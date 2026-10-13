"""Run the standard analysis suite and write a data-backed Markdown report."""

import os
import pandas as pd
from data_utils import run_data_audit
from analysis import build_eda_tables
from statistical_analysis import run_descriptive_statistics, run_rainfall_yield_correlations, run_season_hypothesis_test
from diversification import run_diversification_analysis
from forecasting import get_complete_series_candidates, forecast_crop_production
from visualization import make_eda_plots, plot_diversification_metrics


def _markdown_table(frame, maximum_rows=30):
    """Format a small DataFrame as Markdown without adding a table dependency."""
    if frame is None or frame.empty:
        return "No rows to display."
    shown = frame.head(maximum_rows).copy()
    columns = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in shown.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                text = "N/A"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > maximum_rows:
        lines.append("")
        lines.append(f"Showing {maximum_rows} of {len(frame)} rows; see the CSV output for the full table.")
    return "\n".join(lines)


def _format_number(value, digits=3):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_p_value(value):
    try:
        if pd.isna(value):
            return "N/A"
        if float(value) < 0.001:
            return "<0.001"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def generate_full_report(data, output_root):
    """Generate audit, EDA, statistics, correlations, diversity and forecasts.

    For the forecasting demonstration, one complete annual series with the
    greatest total reported crop area is selected for each state. The selection
    rule is explicit and the selected crops are listed in the report.
    """
    os.makedirs(output_root, exist_ok=True)
    audit, missing, units = run_data_audit(data, output_root)
    tables = build_eda_tables(data, output_root)
    yield_stats, msp_stats = run_descriptive_statistics(data, output_root)
    correlations = run_rainfall_yield_correlations(data, output_root)
    crop_shares, diversity, top_five = run_diversification_analysis(data, output_root)
    plot_paths = make_eda_plots(data, tables, output_root)
    plot_paths.extend(plot_diversification_metrics(diversity, output_root))

    # Perform one season-comparison test for each state's largest-area crop when
    # adequate groups exist. Tests that are not applicable are documented.
    hypothesis_rows = []
    latest_area = tables["crop_area_by_state"]
    for state, group in latest_area.groupby("State"):
        chosen_crop = group.sort_values("Reported_Crop_Area_Ha", ascending=False).iloc[0]["Crop"]
        try:
            hypothesis_rows.append(run_season_hypothesis_test(data, state, chosen_crop, output_root))
        except ValueError as error:
            hypothesis_rows.append({
                "State": state,
                "Crop": chosen_crop,
                "Test": "Kruskal-Wallis H-test",
                "Conclusion": "Skipped: " + str(error),
                "Caveat": "No valid test run because there were insufficient season groups.",
            })
    hypothesis_table = pd.DataFrame(hypothesis_rows)
    hypothesis_table.to_csv(os.path.join(output_root, "tables", "report_hypothesis_tests.csv"), index=False)

    # Select a complete annual state/crop/unit series. If fitting fails for one
    # state, record the reason rather than preventing the rest of the report.
    candidates = get_complete_series_candidates(data)
    forecast_results = []
    if not candidates.empty:
        for state in sorted(data["State"].dropna().unique()):
            state_candidates = candidates[candidates["State"] == state].sort_values(
                "Average_Annual_Reported_Area_Ha", ascending=False
            )
            if state_candidates.empty:
                forecast_results.append({"State": state, "Status": "Skipped: no sufficiently long, complete annual series."})
                continue
            selected = state_candidates.iloc[0]
            try:
                result = forecast_crop_production(
                    data,
                    str(selected["State"]),
                    str(selected["Crop"]),
                    str(selected["Production_Unit"]),
                    output_root,
                )
                forecast_results.append({
                    "State": state,
                    "Crop": result["selected_crop"],
                    "Production_Unit": result["selected_unit"],
                    "Status": "Completed",
                    "ARIMA_MAE": result["arima_mae"],
                    "ARIMA_RMSE": result["arima_rmse"],
                    "Naive_MAE": result["naive_mae"],
                    "Naive_RMSE": result["naive_rmse"],
                })
            except Exception as error:
                forecast_results.append({
                    "State": state,
                    "Crop": str(selected["Crop"]),
                    "Production_Unit": str(selected["Production_Unit"]),
                    "Status": "Forecast failed: " + str(error),
                })
    forecast_table = pd.DataFrame(forecast_results)
    forecast_table.to_csv(os.path.join(output_root, "tables", "report_forecast_summary.csv"), index=False)

    years = data["Year"].astype("string").str.extract(r"(\d{4})", expand=False)
    first_year = data.loc[years.astype(int).idxmin(), "Year"]
    last_year = data.loc[years.astype(int).idxmax(), "Year"]
    states = sorted(data["State"].dropna().unique().tolist())
    units_present = sorted(data["Production_Unit"].dropna().unique().tolist())
    missing_msp = int(data["MSP_Rs_per_Quintal"].isna().sum())

    lines = [
        "# KISAN-X Analysis Report",
        "",
        "## 1. Dataset overview",
        "",
        f"- Observations: **{len(data):,}**",
        f"- Required variables: **{len(data.columns)} columns in the loaded file**",
        f"- Agricultural years: **{first_year} to {last_year}**",
        f"- States: **{', '.join(states)}**",
        f"- Production units: **{', '.join(units_present)}**",
        f"- Missing MSP values: **{missing_msp:,}**; these reflect MSP coverage and were not imputed.",
        "",
        "## 2. Data quality checks",
        "",
        _markdown_table(audit),
        "",
        "## 3. Exploratory patterns",
        "",
        "The tables `area_by_state_year.csv`, `production_by_state_year_unit.csv`, `crop_area_by_state.csv`, `yield_summary_by_state_crop.csv`, and `top_crops_by_state_area.csv` contain the main EDA outputs.",
        "Production is aggregated separately by `Production_Unit`; Tonnes and Bales are never added together.",
        "The area totals are sums of reported crop/season records. They should not be described as unique physical land area, because the same land can be cropped in multiple seasons.",
        "",
        "### Leading crops by cumulative reported area",
        "",
    ]
    for state, group in tables["top_crops"].groupby("State"):
        lines.append(f"**{state}:** " + ", ".join(
            str(row.Crop) for row in group.sort_values("Reported_Crop_Area_Ha", ascending=False).head(3).itertuples()
        ))
        lines.append("")

    lines.extend([
        "## 4. Rainfall and yield relationships",
        "",
        "The correlation output is `rainfall_yield_correlations.csv`. It includes Pearson and Spearman coefficients and p-values for annual and monsoon rainfall versus annual weighted yield for each state/crop/production-unit series with enough observations.",
        "Rainfall values are state-level proxies repeated on district rows. To avoid pseudo-replication, the correlation is calculated only after aggregating to one State-Year-Crop-Unit record.",
        "Correlation is association, not proof that rainfall caused the yield change. Many crop/state combinations are tested, so the strongest findings should be treated as exploratory rather than confirmatory.",
        "",
        "### Largest absolute Pearson associations (exploratory)",
        "",
    ])
    if not correlations.empty:
        for row in correlations.head(5).itertuples():
            lines.append(
                f"- {row.State}, {row.Crop} ({row.Production_Unit}), {row.Rainfall_Measure}: "
                f"r = {_format_number(row.Pearson_r)}, p = {_format_p_value(row.Pearson_p_value)} "
                f"(n = {row.Sample_Years} annual observations)."
            )
    else:
        lines.append("No series met the minimum observation/variation requirements for correlation.")

    lines.extend([
        "",
        "## 5. Crop diversification",
        "",
        "`crop_area_shares.csv` reports the percentage share of each crop in the state's summed reported crop-season area. `diversification_by_state_year.csv` includes:",
        "- **HHI concentration:** the sum of squared crop area shares (0–1). Higher values indicate more concentration in fewer crops.",
        "- **Area diversity indicator:** 1 − HHI. Higher values indicate a more evenly distributed area share.",
        "- **Effective number of crops:** 1 / HHI, an intuitive equivalent count of equally sized crops.",
        "These measures describe reported area shares, not crop profitability or agronomic suitability.",
        "",
        "## 6. Forecasting",
        "",
        "For each state, the report selected the largest-area crop among series that contain at least 15 consecutive annual production observations and one consistent production unit. The simple ARIMA(1,1,0) model was evaluated on the final three observed years and compared with a naive previous-year baseline.",
        "",
    ])
    if not forecast_table.empty:
        lines.append(_markdown_table(forecast_table))
    else:
        lines.append("No forecast series was available.")

    lines.extend([
        "",
        "Forecasts are educational estimates based only on past production. They are not guaranteed outcomes, and future rainfall/MSP are not used as forecast inputs. A short yearly series limits confidence.",
        "",
        "## 7. Seasonal yield hypothesis tests",
        "",
        "The report attempts a Kruskal–Wallis test for the largest-area crop in each state where at least two seasons have five or more yield records. See `report_hypothesis_tests.csv` for the results or skip reasons.",
        "A small p-value at α = 0.05 indicates evidence of differences among the selected season yield distributions; it does not establish causation. The test is exploratory because observations may share districts and years.",
        "",
        "## 8. Output files",
        "",
        "- `tables/`: audit, descriptive summaries, annual trends, correlations, area shares, tests, forecast evaluations.",
        "- `figures/`: labelled Matplotlib/Seaborn charts, including production trends and forecast plots.",
        "- `forecasts/`: holdout evaluation and future-year predictions for each selected state series.",
        "",
        "## 9. Main limitations",
        "",
        "1. Rainfall is an IMD subdivision-derived state-level proxy, not district-level rainfall.",
        "2. MSP is available only for matched crop-years; missing MSP is not zero.",
        "3. Production units differ and must remain separated in totals and forecasts.",
        "4. Area sums are reported crop-season planted-area records, not unique farmland area.",
        "5. Correlation is not causation; the data do not include complete input costs, irrigation status, soil properties, or market demand.",
        "6. Forecast accuracy cannot be guaranteed and performance on one short holdout period is uncertain.",
        "",
        "---",
        "Generated by KISAN-X CLI. Read `docs/PROJECT_FLOW.md` and `docs/VIVA_GUIDE.md` before interpreting the results.",
    ])

    report_path = os.path.join(output_root, "KISAN_X_ANALYSIS_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")

    return {
        "report_path": report_path,
        "audit": audit,
        "eda_tables": tables,
        "yield_statistics": yield_stats,
        "msp_statistics": msp_stats,
        "correlations": correlations,
        "diversification": diversity,
        "forecast_summary": forecast_table,
        "hypothesis_tests": hypothesis_table,
        "plots": plot_paths,
    }
