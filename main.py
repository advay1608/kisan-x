"""Entry point for the KISAN-X menu-driven command-line application."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "src")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from data_utils import load_dataset, make_output_directories, run_data_audit
from analysis import build_eda_tables
from statistical_analysis import (
    run_descriptive_statistics,
    run_rainfall_yield_correlations,
    run_season_hypothesis_test,
)
from diversification import run_diversification_analysis
from forecasting import get_complete_series_candidates, forecast_crop_production
from visualization import make_eda_plots, plot_diversification_metrics, plot_rainfall_yield_scatter
from report import generate_full_report


def print_header():
    print("\n" + "=" * 62)
    print("                         KISAN-X")
    print("       CROP PRODUCTION ANALYSIS AND FORECASTING SYSTEM")
    print("=" * 62)


def print_menu():
    print("\nMAIN MENU")
    print("1. Dataset overview and data-quality audit")
    print("2. Exploratory analysis and charts")
    print("3. Descriptive statistical summaries")
    print("4. Rainfall-yield relationships")
    print("5. Compare yield across seasons (hypothesis test)")
    print("6. Crop diversification analysis")
    print("7. Forecast selected crop production")
    print("8. Generate complete analysis report")
    print("0. Exit")


def show_table(table, rows=15):
    if table is None or table.empty:
        print("No results are available for this selection.")
        return
    print(table.head(rows).to_string(index=False))
    if len(table) > rows:
        print("... showing first", rows, "of", len(table), "rows. See the saved CSV for all results.")


def choose_from_list(prompt, values):
    """Ask for a value and match it case-insensitively to known labels."""
    values = [str(value) for value in values]
    if not values:
        raise ValueError("No available choices were found for this operation.")
    lookup = {value.casefold(): value for value in values}
    print("\nAvailable options:")
    for index, value in enumerate(values, start=1):
        print(str(index).rjust(3) + ". " + value)
    answer = input(prompt + " (enter name or number): ").strip()
    if answer.isdigit():
        position = int(answer)
        if 1 <= position <= len(values):
            return values[position - 1]
    if answer.casefold() in lookup:
        return lookup[answer.casefold()]
    raise ValueError("Selection not recognized. Please choose a listed name or number.")


def audit_option(data, output_root):
    audit, missing, units = run_data_audit(data, output_root)
    print("\nDATA AUDIT SUMMARY")
    show_table(audit, rows=30)
    print("\nMISSING VALUES BY COLUMN")
    show_table(missing, rows=20)
    print("\nPRODUCTION UNIT COVERAGE")
    show_table(units, rows=20)
    print("\nSaved audit tables to outputs/tables/.")


def eda_option(data, output_root):
    tables = build_eda_tables(data, output_root)
    plots = make_eda_plots(data, tables, output_root)
    print("\nTOP CROPS BY REPORTED AREA (ALL YEARS COMBINED)")
    show_table(tables["top_crops"], rows=20)
    print("\nSTATE/YEAR PRODUCTION TOTALS (KEPT SEPARATE BY UNIT)")
    show_table(tables["production_by_state_year_unit"], rows=20)
    print("\nSaved", len(plots), "figures to outputs/figures/ and summary tables to outputs/tables/.")
    print("Reminder: reported area is summed across crop-season records, not unique physical land.")


def descriptive_option(data, output_root):
    yield_stats, msp_stats = run_descriptive_statistics(data, output_root)
    print("\nYIELD DESCRIPTIVE STATISTICS BY STATE / CROP / UNIT")
    show_table(yield_stats, rows=25)
    print("\nMSP DESCRIPTIVE STATISTICS (UNIQUE CROP-YEAR VALUES)")
    show_table(msp_stats, rows=20)
    print("\nSaved full tables to outputs/tables/.")


def relationships_option(data, output_root):
    results = run_rainfall_yield_correlations(data, output_root)
    if results.empty:
        print("No series met the minimum sample-size and variation requirements.")
        return
    print("\nSTRONGEST ABSOLUTE PEARSON ASSOCIATIONS (EXPLORATORY)")
    show_table(results.drop(columns=["Absolute_Pearson_r"], errors="ignore"), rows=12)
    print("\nOptional: create a rainfall-yield scatter plot for one state/crop.")
    answer = input("Create a scatter plot? (y/n): ").strip().lower()
    if answer == "y":
        state = choose_from_list("Choose a state", sorted(data["State"].dropna().unique()))
        crops = sorted(data.loc[data["State"] == state, "Crop"].dropna().unique())
        crop = choose_from_list("Choose a crop", crops)
        path = plot_rainfall_yield_scatter(data, state, crop, output_root)
        print("Saved scatter plot:", path)
    print("\nInterpretation: association does not prove causation; rainfall is a state-level proxy.")


def hypothesis_option(data, output_root):
    state = choose_from_list("Choose a state", sorted(data["State"].dropna().unique()))
    crops = sorted(data.loc[data["State"] == state, "Crop"].dropna().unique())
    crop = choose_from_list("Choose a crop", crops)
    result = run_season_hypothesis_test(data, state, crop, output_root)
    print("\nKRUSKAL-WALLIS SEASON COMPARISON")
    for key, value in result.items():
        print(key.replace("_", " ") + ": " + str(value))
    print("\nThe test compares distributions; it does not show that season alone caused the differences.")
    print("Saved to outputs/tables/latest_season_hypothesis_test.csv.")


def diversification_option(data, output_root):
    shares, metrics, top_five = run_diversification_analysis(data, output_root)
    paths = plot_diversification_metrics(metrics, output_root)
    last_year = metrics["Year_Start"].max()
    latest = metrics[metrics["Year_Start"] == last_year].sort_values("State")
    print("\nCROP DIVERSIFICATION METRICS FOR THE LATEST AVAILABLE YEAR")
    show_table(latest, rows=20)
    print("\nTOP FIVE CROP AREA SHARES FOR THE LATEST YEAR")
    latest_shares = shares[shares["Year_Start"] == last_year].sort_values(
        ["State", "Area_Share_Percent"], ascending=[True, False]
    ).groupby("State", as_index=False).head(5)
    show_table(latest_shares, rows=20)
    print("\nSaved tables and", len(paths), "diversification charts.")
    print("HHI = sum of squared area shares; higher HHI means greater concentration.")


def forecasting_option(data, output_root):
    candidates = get_complete_series_candidates(data)
    if candidates.empty:
        print("No crop/state/unit series has enough consecutive annual observations to forecast.")
        return
    state = choose_from_list("Choose a state", sorted(candidates["State"].unique()))
    state_candidates = candidates[candidates["State"] == state].sort_values(
        "Average_Annual_Reported_Area_Ha", ascending=False
    )
    print("\nEligible series have at least 15 consecutive annual observations.")
    print(state_candidates[["Crop", "Production_Unit", "Annual_Points", "First_Year", "Last_Year"]].to_string(index=False))
    crop = choose_from_list("Choose a crop from the table above", sorted(state_candidates["Crop"].unique()))
    crop_candidates = state_candidates[state_candidates["Crop"] == crop]
    units = sorted(crop_candidates["Production_Unit"].unique())
    unit = units[0] if len(units) == 1 else choose_from_list("Choose production unit", units)

    result = forecast_crop_production(data, state, crop, unit, output_root)
    print("\nHOLDOUT EVALUATION (LOWER MAE/RMSE IS BETTER)")
    show_table(result["evaluation"], rows=10)
    print("\nFUTURE FORECAST")
    show_table(result["future_forecast"], rows=10)
    if result["arima_mae"] < result["naive_mae"]:
        print("On this holdout, ARIMA had lower MAE than the naive baseline.")
    else:
        print("On this holdout, ARIMA did not improve MAE over the naive baseline.")
    print("Forecast is an estimate from past production only; it is not a guarantee.")


def full_report_option(data, output_root):
    print("Running audit, EDA, statistical summaries, correlations, diversification, charts, and selected forecasts...")
    result = generate_full_report(data, output_root)
    print("\nREPORT CREATED:", result["report_path"])
    print("\nFORECAST SUMMARY")
    show_table(result["forecast_summary"], rows=12)
    print("\nThe report and all supporting tables/figures are saved in the outputs folder.")


def main():
    print_header()
    data_path = os.path.join(PROJECT_ROOT, "data", "this_file.csv")
    output_root = make_output_directories(PROJECT_ROOT)
    try:
        data = load_dataset(data_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        print("\nUnable to load the project dataset:")
        print(error)
        return

    print("Loaded dataset:", data.shape[0], "rows x", data.shape[1], "columns")
    print("Source file:", os.path.relpath(data_path, PROJECT_ROOT))
    print("The input CSV is not modified by KISAN-X.")

    actions = {
        "1": audit_option,
        "2": eda_option,
        "3": descriptive_option,
        "4": relationships_option,
        "5": hypothesis_option,
        "6": diversification_option,
        "7": forecasting_option,
        "8": full_report_option,
    }

    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        if choice == "0":
            print("Thank you for using KISAN-X. Goodbye!")
            break
        if choice not in actions:
            print("Invalid choice. Enter a number from 0 to 8.")
            continue
        try:
            actions[choice](data, output_root)
        except (ValueError, FileNotFoundError, OSError) as error:
            print("\nOperation could not be completed:", error)
        except Exception as error:
            # Unexpected details are shown for debugging while keeping the
            # menu alive; do not silently hide a computation failure.
            print("\nUnexpected error:", type(error).__name__, str(error))
            print("Check the error details before interpreting any output.")


if __name__ == "__main__":
    main()
