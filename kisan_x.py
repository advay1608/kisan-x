"""KISAN-X analysis module.

All data loading, analysis, charting and reporting functions live here.
`main.py` only provides the menu and calls these functions.

Scope reminders that apply to every function below:
  * The source CSV is read-only and is never modified.
  * Production is never summed across different Production_Unit values.
  * Area totals are sums of reported crop-season records, not unique land.
  * Rainfall is a state-level proxy repeated on district rows.
  * Correlation is association, never proof of causation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid", context="notebook")

# ---------------------------------------------------------------------------
# Schema and project paths (EX 5: os paths, EX 8: Pandas)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Year", "State", "District", "Crop", "Season", "Area_Ha",
    "Production", "Production_Unit", "Yield", "Annual_Rainfall_mm",
    "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal",
]
NUMERIC_COLUMNS = [
    "Area_Ha", "Production", "Yield", "Annual_Rainfall_mm",
    "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal",
]
TEXT_COLUMNS = ["Year", "State", "District", "Crop", "Season", "Production_Unit"]
KEY_COLUMNS = ["Year", "State", "District", "Crop", "Season"]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "this_file.csv")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")

MIN_FORECAST_YEARS = 15          # shortest annual series we are willing to model
MIN_CORRELATION_PAIRS = 8        # shortest annual series we are willing to correlate
MIN_SEASON_OBSERVATIONS = 5      # smallest season group kept in the Kruskal-Wallis test


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def make_output_directories():
    """Create outputs/tables, outputs/figures and outputs/forecasts if missing."""
    for folder in ["tables", "figures", "forecasts"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)
    return OUTPUT_ROOT


def save_table(frame, filename, folder="tables"):
    """Save a DataFrame as CSV under outputs/<folder>/ and return its path."""
    directory = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    frame.to_csv(path, index=False)
    return path


def save_figure(figure, filename):
    """Save a Matplotlib figure under outputs/figures/ and close it."""
    directory = os.path.join(OUTPUT_ROOT, "figures")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def safe_name(*parts):
    """Build a file-safe name from label parts, e.g. Madhya_Pradesh_Wheat_Tonnes."""
    text = "_".join(str(part) for part in parts)
    return text.replace("/", "-").replace(" ", "_")


# ---------------------------------------------------------------------------
# Loading and shared preparation
# ---------------------------------------------------------------------------

def load_data(file_path=DATA_PATH):
    """Read the project CSV and check the 12 required columns.

    Text that cannot be read as a number becomes NaN and is reported by the
    audit. Nothing is dropped, imputed or written back to the source file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Dataset not found at: " + file_path +
            "\nPlace the final CSV at data/this_file.csv."
        )

    data = pd.read_csv(file_path, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns) +
            "\nExpected schema: " + ", ".join(REQUIRED_COLUMNS)
        )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    # Year_Start turns "1997-98" into 1997 so years can be sorted and modelled.
    data["Year_Start"] = pd.to_numeric(
        data["Year"].str.extract(r"(\d{4})", expand=False), errors="coerce"
    )
    return data


def annual_series(data):
    """Collapse district/season rows to one row per Year-State-Crop-Unit.

    Rainfall is a state-level proxy copied onto every district row, so "first"
    keeps one value per state-year instead of counting each district copy as an
    independent rainfall observation. Weighted yield is total production over
    total area, which is valid here because both come from the same unit group.
    """
    grouped = data.groupby(
        ["Year_Start", "Year", "State", "Crop", "Production_Unit"], as_index=False
    ).agg(
        Area_Ha=("Area_Ha", "sum"),
        Production=("Production", "sum"),
        Annual_Rainfall_mm=("Annual_Rainfall_mm", "first"),
        JJAS_Rainfall_mm=("JJAS_Rainfall_mm", "first"),
    )
    grouped["Weighted_Yield"] = (
        grouped["Production"] / grouped["Area_Ha"].replace(0, np.nan)
    )
    return grouped.sort_values(["State", "Crop", "Production_Unit", "Year_Start"])


# ---------------------------------------------------------------------------
# 1. Data audit (EX 8: missing-value summaries)
# ---------------------------------------------------------------------------

def audit_data(data):
    """Report shape, duplicates, missingness, units and plausibility checks.

    Missing MSP is an expected coverage gap: it is counted and kept, never
    filled with zero or a mean. Duplicates are counted, never silently deleted.
    """
    # Yield should equal Production / Area; compare with a float tolerance.
    derived_yield = data["Production"] / data["Area_Ha"].replace(0, np.nan)
    yield_mismatch = int((~np.isclose(
        derived_yield.to_numpy(dtype=float),
        data["Yield"].to_numpy(dtype=float),
        rtol=1e-7, atol=1e-7, equal_nan=True,
    )).sum())

    # Rainfall should hold exactly one value per State + Year in this dataset.
    rainfall_spread = data.groupby(["State", "Year_Start"])[
        ["Annual_Rainfall_mm", "JJAS_Rainfall_mm"]
    ].nunique(dropna=True)
    rainfall_conflicts = int((rainfall_spread > 1).any(axis=1).sum())

    rows = [
        ("Rows", len(data), "Observations loaded"),
        ("Columns", len(data.columns) - 1, "Required columns (Year_Start is derived in memory)"),
        ("Exact duplicate rows", int(data.duplicated().sum()), "Expected 0; counted, not deleted"),
        ("Duplicate analytical keys", int(data.duplicated(KEY_COLUMNS).sum()),
         "Key: Year + State + District + Crop + Season; expected 0"),
        ("States", int(data["State"].nunique()), "Unique states represented"),
        ("Districts", int(data["District"].nunique()), "Unique district labels"),
        ("Crops", int(data["Crop"].nunique()), "Unique crop labels"),
        ("Seasons", int(data["Season"].nunique()), "Unique crop-season labels"),
        ("Production units", int(data["Production_Unit"].nunique()),
         "Totals are only ever formed within one unit"),
        ("First agricultural year", str(data.loc[data["Year_Start"].idxmin(), "Year"]),
         "Earliest year label"),
        ("Last agricultural year", str(data.loc[data["Year_Start"].idxmax(), "Year"]),
         "Latest year label"),
        ("Missing MSP values", int(data["MSP_Rs_per_Quintal"].isna().sum()),
         "Expected coverage gap; not replaced with zero or the mean"),
        ("Negative area values", int((data["Area_Ha"] < 0).sum()), "Expected 0"),
        ("Negative production values", int((data["Production"] < 0).sum()), "Expected 0"),
        ("Negative yield values", int((data["Yield"] < 0).sum()), "Expected 0"),
        ("Yield inconsistent with Production / Area", yield_mismatch,
         "Tolerance 1e-7; investigate if non-zero"),
        ("State-year rainfall conflicts", rainfall_conflicts,
         "Expected 0: the proxy should hold one value per State + Year"),
    ]
    audit = pd.DataFrame(rows, columns=["Metric", "Value", "Interpretation"])
    save_table(audit, "data_audit_summary.csv")

    missing = pd.DataFrame({
        "Column": list(data.columns),
        "Missing_Count": [int(data[c].isna().sum()) for c in data.columns],
    })
    missing["Missing_Percent"] = (missing["Missing_Count"] / len(data) * 100).round(4)
    missing["Note"] = np.where(
        missing["Column"] == "MSP_Rs_per_Quintal",
        "Expected coverage gap", "Review if non-zero",
    )
    missing = missing.sort_values("Missing_Count", ascending=False)
    save_table(missing, "missing_values_by_column.csv")

    units = data.groupby("Production_Unit", dropna=False, as_index=False).agg(
        Records=("Production", "size"),
        Distinct_Crops=("Crop", "nunique"),
        Total_Reported_Area_Ha=("Area_Ha", "sum"),
    )
    save_table(units, "production_unit_coverage.csv")
    return audit, missing, units


# ---------------------------------------------------------------------------
# 2. Exploratory tables and charts (EX 8, EX 9, EX 10)
# ---------------------------------------------------------------------------

def build_eda_tables(data):
    """Build the area, production and crop-ranking summaries used elsewhere."""
    # Sum of reported crop-season area across district records. The same field
    # can appear under several crops or seasons, so this is NOT unique land.
    area_by_state_year = data.groupby(
        ["Year_Start", "Year", "State"], as_index=False
    ).agg(Reported_Crop_Area_Ha=("Area_Ha", "sum"), Records=("Area_Ha", "size"))
    save_table(area_by_state_year, "area_by_state_year.csv")

    # Production is kept split by unit so Tonnes and Bales are never added.
    production_by_state_year = data.groupby(
        ["Year_Start", "Year", "State", "Production_Unit"], as_index=False
    ).agg(Production_Total=("Production", "sum"), Records=("Production", "size"))
    save_table(production_by_state_year, "production_by_state_year_unit.csv")

    crop_area_by_state = data.groupby(["State", "Crop"], as_index=False).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"), Records=("Area_Ha", "size"),
    ).sort_values(["State", "Reported_Crop_Area_Ha"], ascending=[True, False])
    save_table(crop_area_by_state, "crop_area_by_state.csv")

    top_crops = crop_area_by_state.groupby("State", as_index=False).head(10).copy()
    top_crops["Rank_By_Reported_Area"] = top_crops.groupby("State").cumcount() + 1
    save_table(top_crops, "top_crops_by_state_area.csv")

    return {
        "area_by_state_year": area_by_state_year,
        "production_by_state_year": production_by_state_year,
        "crop_area_by_state": crop_area_by_state,
        "top_crops": top_crops,
    }


def make_eda_charts(data, tables):
    """Save the line, bar and histogram charts for the exploratory section."""
    paths = []

    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=tables["area_by_state_year"], x="Year_Start",
                 y="Reported_Crop_Area_Ha", hue="State", marker="o", ax=axis)
    axis.set_title("Reported Crop Area by State and Agricultural Year")
    axis.set_xlabel("Agricultural year (starting year)")
    axis.set_ylabel("Sum of reported crop-season area (ha)")
    paths.append(save_figure(figure, "reported_crop_area_trends_by_state.png"))

    # One production chart per unit, because units cannot share one total axis.
    for unit, group in tables["production_by_state_year"].groupby("Production_Unit"):
        figure, axis = plt.subplots(figsize=(10, 5.5))
        sns.lineplot(data=group, x="Year_Start", y="Production_Total",
                     hue="State", marker="o", ax=axis)
        axis.set_title("Reported Production Trend by State (unit: " + str(unit) + ")")
        axis.set_xlabel("Agricultural year (starting year)")
        axis.set_ylabel("Production (" + str(unit) + ")")
        paths.append(save_figure(figure, "production_trends_" + safe_name(unit) + ".png"))

    for state, group in tables["top_crops"].groupby("State"):
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.barplot(data=group.sort_values("Reported_Crop_Area_Ha"),
                    x="Reported_Crop_Area_Ha", y="Crop", color="steelblue", ax=axis)
        axis.set_title("Top Crops by Reported Area - " + str(state))
        axis.set_xlabel("Sum of reported crop-season area (ha)")
        axis.set_ylabel("Crop")
        paths.append(save_figure(figure, "top_crops_area_" + safe_name(state) + ".png"))

    tonne_yield = data.loc[
        (data["Production_Unit"] == "Tonnes") & data["Yield"].notna(), "Yield"
    ]
    if len(tonne_yield):
        figure, axis = plt.subplots(figsize=(9, 5.5))
        sns.histplot(tonne_yield, bins=40, kde=True, color="steelblue", ax=axis)
        axis.set_title("Yield Distribution for Records Measured in Tonnes")
        axis.set_xlabel("Yield (tonnes per hectare)")
        axis.set_ylabel("Record count")
        paths.append(save_figure(figure, "yield_distribution_tonnes.png"))

    return paths


# ---------------------------------------------------------------------------
# 3. Descriptive statistics (EX 7 NumPy, EX 11 SciPy)
# ---------------------------------------------------------------------------

def descriptive_statistics(data):
    """Describe yield per State/Crop/Unit group and MSP per crop.

    Yield groups keep the production unit separate so tonnes-per-hectare and
    bales-per-hectare are never mixed inside one mean.
    """
    rows = []
    for (state, crop, unit), group in data.groupby(["State", "Crop", "Production_Unit"]):
        values = group["Yield"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        # NumPy for the basic moments, SciPy for skewness and kurtosis.
        description = stats.describe(values, ddof=1, bias=False) if len(values) >= 2 else None
        rows.append({
            "State": state, "Crop": crop, "Production_Unit": unit,
            "Count": len(values),
            "Mean_Yield": float(np.mean(values)),
            "Median_Yield": float(np.median(values)),
            "Standard_Deviation": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
            "Variance": float(description.variance) if description else np.nan,
            "Skewness": float(description.skewness) if len(values) >= 3 else np.nan,
            "Kurtosis": float(description.kurtosis) if len(values) >= 4 else np.nan,
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
        })
    yield_stats = pd.DataFrame(rows).sort_values(["State", "Crop", "Production_Unit"])
    save_table(yield_stats, "descriptive_yield_statistics.csv")

    # MSP is a crop-year price repeated on every matching district row, so the
    # duplicates are removed for this summary only. The dataset itself is intact.
    msp = data[["Year", "Crop", "MSP_Rs_per_Quintal"]].dropna().drop_duplicates()
    msp_stats = msp.groupby("Crop", as_index=False).agg(
        Distinct_Crop_Years=("Year", "nunique"),
        Mean_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "mean"),
        Minimum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "min"),
        Maximum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "max"),
    )
    save_table(msp_stats, "descriptive_msp_statistics.csv")
    return yield_stats, msp_stats


# ---------------------------------------------------------------------------
# 4. Rainfall and yield association (EX 11 correlation)
# ---------------------------------------------------------------------------

def rainfall_yield_correlations(data):
    """Correlate rainfall with weighted yield on aggregated annual records.

    The aggregation in annual_series() happens first on purpose: without it,
    every district row would repeat the same state rainfall value and inflate
    the sample size. The result is an association, not a causal effect.
    """
    annual = annual_series(data)
    rows = []
    for (state, crop, unit), group in annual.groupby(["State", "Crop", "Production_Unit"]):
        for column, label in [("Annual_Rainfall_mm", "Annual rainfall"),
                              ("JJAS_Rainfall_mm", "Monsoon (JJAS) rainfall")]:
            pair = group[[column, "Weighted_Yield"]].dropna()
            if len(pair) < MIN_CORRELATION_PAIRS:
                continue
            rainfall = pair[column].to_numpy(dtype=float)
            yields = pair["Weighted_Yield"].to_numpy(dtype=float)
            if np.std(rainfall) == 0 or np.std(yields) == 0:
                continue        # a flat series has no correlation to report
            pearson = stats.pearsonr(rainfall, yields)
            spearman = stats.spearmanr(rainfall, yields)
            rows.append({
                "State": state, "Crop": crop, "Production_Unit": unit,
                "Rainfall_Measure": label,
                "Sample_Years": len(pair),
                "Pearson_r": float(pearson.statistic),
                "Pearson_p_value": float(pearson.pvalue),
                "Spearman_rho": float(spearman.statistic),
                "Spearman_p_value": float(spearman.pvalue),
                "Interpretation_Guide": "Association only; not proof of causation.",
            })

    results = pd.DataFrame(rows)
    if not results.empty:
        results["Absolute_Pearson_r"] = results["Pearson_r"].abs()
        results = results.sort_values("Absolute_Pearson_r", ascending=False)
    save_table(results, "rainfall_yield_correlations.csv")
    return results


def plot_rainfall_yield(data, state, crop):
    """Scatter monsoon rainfall against weighted yield for one state and crop."""
    annual = annual_series(data)
    subset = annual[(annual["State"] == state) & (annual["Crop"] == crop)]
    if subset.empty:
        raise ValueError("No annual records found for " + state + " / " + crop + ".")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=subset, x="JJAS_Rainfall_mm", y="Weighted_Yield",
                    hue="Production_Unit", s=75, ax=axis)
    axis.set_title("Monsoon Rainfall and Crop Yield - " + state + " / " + crop)
    axis.set_xlabel("JJAS rainfall proxy (mm)")
    axis.set_ylabel("Weighted yield (production unit per hectare)")
    return save_figure(figure, "rainfall_yield_" + safe_name(state, crop) + ".png")


# ---------------------------------------------------------------------------
# 5. Season hypothesis test (EX 11 Kruskal-Wallis)
# ---------------------------------------------------------------------------

def season_yield_test(data, state, crop, alpha=0.05):
    """Compare yield distributions across seasons for one state and crop.

    Kruskal-Wallis is non-parametric, so it does not assume normally
    distributed yields. Records share districts and years, so this is
    exploratory: a small p-value never proves that the season caused it.
    """
    subset = data[(data["State"] == state) & (data["Crop"] == crop)]
    groups = {}
    for season, group in subset.groupby("Season"):
        values = group["Yield"].dropna().to_numpy(dtype=float)
        if len(values) >= MIN_SEASON_OBSERVATIONS:
            groups[str(season)] = values

    if len(groups) < 2:
        raise ValueError(
            "Not enough data for a season comparison. Choose a crop with at least two "
            "seasons that each have " + str(MIN_SEASON_OBSERVATIONS) + " or more yield values."
        )

    result = stats.kruskal(*groups.values())
    p_value = float(result.pvalue)
    summary = {
        "State": state,
        "Crop": crop,
        "Test": "Kruskal-Wallis H-test",
        "Seasons_Compared": ", ".join(groups),
        "Observations_Per_Season": "; ".join(s + ": " + str(len(v)) for s, v in groups.items()),
        "H_Statistic": float(result.statistic),
        "P_Value": p_value,
        "Alpha": alpha,
        "Reject_Null": bool(p_value < alpha),
        "Null_Hypothesis": "Yield distributions are the same across the selected seasons.",
        "Conclusion": ("Evidence of a difference at the chosen significance level."
                       if p_value < alpha else
                       "Insufficient evidence of a difference at the chosen significance level."),
        "Caveat": "Exploratory; records share districts and years, so no causal claim.",
    }
    save_table(pd.DataFrame([summary]), "latest_season_hypothesis_test.csv")
    return summary


# ---------------------------------------------------------------------------
# 6. Crop diversification (EX 7 vectorized arithmetic)
# ---------------------------------------------------------------------------

def diversification(data):
    """Compute crop-area shares and concentration measures per state and year.

    HHI is the sum of squared crop area shares (0 to 1). Higher means the area
    is concentrated in fewer crops. 1 - HHI is a diversity indicator and 1 / HHI
    is the effective number of equally sized crops. Shares describe reported
    area only; they say nothing about profit or agronomic suitability.
    """
    crop_area = data.groupby(["Year_Start", "Year", "State", "Crop"], as_index=False).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum")
    )
    state_totals = crop_area.groupby(["Year_Start", "State"])[
        "Reported_Crop_Area_Ha"
    ].transform("sum")
    crop_area["Area_Share"] = crop_area["Reported_Crop_Area_Ha"] / state_totals
    crop_area["Area_Share_Percent"] = crop_area["Area_Share"] * 100
    save_table(crop_area, "crop_area_shares.csv")

    crop_area["Squared_Share"] = crop_area["Area_Share"] ** 2
    metrics = crop_area.groupby(["Year_Start", "Year", "State"], as_index=False).agg(
        Number_of_Crops=("Crop", "nunique"),
        HHI_Area_Concentration=("Squared_Share", "sum"),
        Total_Reported_Crop_Area_Ha=("Reported_Crop_Area_Ha", "sum"),
    )
    metrics["Area_Diversity_Index"] = 1 - metrics["HHI_Area_Concentration"]
    metrics["Effective_Number_of_Crops"] = 1 / metrics["HHI_Area_Concentration"].replace(0, np.nan)
    save_table(metrics, "diversification_by_state_year.csv")

    top_five = crop_area.sort_values(
        ["State", "Year_Start", "Area_Share_Percent"], ascending=[True, True, False]
    ).groupby(["Year_Start", "State"], as_index=False).head(5)
    save_table(top_five, "top_five_crop_shares_by_year.csv")
    return crop_area, metrics, top_five


def plot_diversification(metrics):
    """Chart the diversity indicator and the HHI concentration by state/year."""
    paths = []
    for column, title, ylabel, filename in [
        ("Area_Diversity_Index", "Crop-Area Diversity by State and Year",
         "1 - HHI (higher means more even shares)", "area_diversity_by_state.png"),
        ("HHI_Area_Concentration", "Crop-Area Concentration by State and Year",
         "HHI (higher means more concentrated)", "area_concentration_by_state.png"),
    ]:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        sns.lineplot(data=metrics, x="Year_Start", y=column, hue="State", marker="o", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Agricultural year (starting year)")
        axis.set_ylabel(ylabel)
        paths.append(save_figure(figure, filename))
    return paths


# ---------------------------------------------------------------------------
# 7. Forecasting (stationarity, differencing, ARIMA, holdout evaluation)
# ---------------------------------------------------------------------------

def forecast_candidates(data):
    """List State/Crop/Unit series with enough consecutive annual observations.

    Series with gaps are excluded rather than patched: filling a missing year
    with zero would invent a crop failure that the data never recorded.
    """
    annual = data.groupby(
        ["Year_Start", "State", "Crop", "Production_Unit"], as_index=False
    ).agg(Production=("Production", "sum"), Reported_Crop_Area_Ha=("Area_Ha", "sum"))

    rows = []
    for (state, crop, unit), group in annual.groupby(["State", "Crop", "Production_Unit"]):
        years = np.sort(group["Year_Start"].to_numpy(dtype=int))
        if len(years) < MIN_FORECAST_YEARS or not np.all(np.diff(years) == 1):
            continue
        rows.append({
            "State": state, "Crop": crop, "Production_Unit": unit,
            "Annual_Points": len(years),
            "First_Year": int(years[0]), "Last_Year": int(years[-1]),
            "Average_Annual_Reported_Area_Ha": float(group["Reported_Crop_Area_Ha"].mean()),
        })

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["State", "Average_Annual_Reported_Area_Ha"], ascending=[True, False]
        )
    return candidates


def year_label(year_start):
    """Format 1997 as the agricultural-year label 1997-98."""
    return str(int(year_start)) + "-" + str((int(year_start) + 1) % 100).zfill(2)


def error_metrics(actual, predicted):
    """Return mean absolute error and root mean squared error (NumPy)."""
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def production_series(data, state, crop, unit):
    """Return the consecutive annual production series for one state/crop/unit."""
    subset = data[(data["State"] == state) & (data["Crop"] == crop) &
                  (data["Production_Unit"] == unit)]
    if subset.empty:
        raise ValueError("No records found for the selected state, crop and unit.")

    annual = subset.groupby(["Year_Start", "Year"], as_index=False).agg(
        Production=("Production", "sum")
    ).sort_values("Year_Start")

    years = annual["Year_Start"].to_numpy(dtype=int)
    if len(years) < MIN_FORECAST_YEARS:
        raise ValueError("At least " + str(MIN_FORECAST_YEARS) + " annual points are required.")
    if not np.all(np.diff(years) == 1):
        raise ValueError(
            "This series has missing years. KISAN-X does not fill gaps with zero; "
            "choose a complete series instead."
        )
    if annual["Production"].nunique() < 3:
        raise ValueError("This series has too little variation to forecast usefully.")
    return annual


def forecast_production(data, state, crop, unit, holdout_years=3, future_years=3):
    """Evaluate ARIMA(1,1,0) against a naive baseline, then forecast ahead.

    The order (1,1,0) is fixed and explainable - one autoregressive term on the
    once-differenced series - rather than chosen by an automatic model search.
    The last few observed years are held out chronologically (never shuffled,
    because that would let the model see the future) and both models predict
    them. Only after that is the model refitted on the whole series.
    """
    annual = production_series(data, state, crop, unit)
    values = annual["Production"].to_numpy(dtype=float)
    years = annual["Year_Start"].to_numpy(dtype=int)

    # Keep at least 12 training years whatever holdout length is requested.
    holdout_years = min(max(2, int(holdout_years)), len(values) - 12)
    train, test = values[:-holdout_years], values[-holdout_years:]
    test_labels = annual["Year"].iloc[-holdout_years:].astype(str).tolist()

    # Augmented Dickey-Fuller on the training part only: a large p-value means
    # we cannot reject a unit root, which is why the model differences (d = 1).
    adf_statistic, adf_p_value = np.nan, np.nan
    adf_note = "Stationarity test unavailable."
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(train, autolag="AIC")
        adf_statistic, adf_p_value = float(adf[0]), float(adf[1])
        adf_note = ("Evidence against a unit root at 5%."
                    if adf_p_value < 0.05 else
                    "No evidence against a unit root at 5%; differencing is justified.")
    except Exception as error:
        adf_note = "Stationarity test unavailable: " + str(error)

    # Baseline: predict each holdout year with the previous observed year.
    naive_predictions = np.concatenate(([train[-1]], test[:-1]))
    naive_mae, naive_rmse = error_metrics(test, naive_predictions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holdout_model = ARIMA(train, order=(1, 1, 0), enforce_stationarity=False,
                              enforce_invertibility=False).fit()
        arima_predictions = np.asarray(holdout_model.forecast(steps=holdout_years), dtype=float)

        final_model = ARIMA(values, order=(1, 1, 0), enforce_stationarity=False,
                            enforce_invertibility=False).fit()
        future = final_model.get_forecast(steps=future_years)
        future_values = np.asarray(future.predicted_mean, dtype=float)
        interval = np.asarray(future.conf_int(alpha=0.05), dtype=float)
    arima_mae, arima_rmse = error_metrics(test, arima_predictions)

    future_year_starts = np.arange(years[-1] + 1, years[-1] + 1 + future_years)
    future_table = pd.DataFrame([{
        "State": state, "Crop": crop, "Production_Unit": unit,
        "Forecast_Year": year_label(future_year),
        # The raw model number is kept beside the floored value so the
        # "production cannot be negative" correction stays visible.
        "ARIMA_Raw_Forecast": float(future_values[index]),
        "Forecast_Production_NonNegative": float(max(0.0, future_values[index])),
        "Lower_95_Percent": float(interval[index, 0]),
        "Upper_95_Percent": float(interval[index, 1]),
        "Model": "ARIMA(1,1,0)",
        "Note": "Estimate from past production only; rainfall and MSP are not inputs.",
    } for index, future_year in enumerate(future_year_starts)])

    evaluation = pd.DataFrame([{
        "State": state, "Crop": crop, "Production_Unit": unit,
        "Model": label,
        "Training_Years": len(train), "Holdout_Years": len(test),
        "Holdout_First_Year": test_labels[0], "Holdout_Last_Year": test_labels[-1],
        "MAE": mae, "RMSE": rmse,
        "ADF_Statistic_Training": adf_statistic,
        "ADF_P_Value_Training": adf_p_value,
        "Stationarity_Note": adf_note,
    } for label, mae, rmse in [("Naive previous-year baseline", naive_mae, naive_rmse),
                               ("ARIMA(1,1,0)", arima_mae, arima_rmse)]])

    name = safe_name(state, crop, unit)
    save_table(future_table, name + "_future_forecast.csv", folder="forecasts")
    save_table(evaluation, name + "_evaluation.csv", folder="forecasts")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(years[:-holdout_years], train, marker="o", label="Training observations")
    axis.plot(years[-holdout_years:], test, marker="o", linestyle="--", label="Actual holdout")
    axis.plot(years[-holdout_years:], arima_predictions, marker="x", linestyle="--",
              label="ARIMA holdout forecast")
    axis.plot(future_year_starts, future_values, marker="o", linestyle=":",
              label="Future ARIMA forecast")
    axis.set_title("Crop Production Forecast - " + state + " / " + crop)
    axis.set_xlabel("Agricultural year (starting year)")
    axis.set_ylabel("Production (" + unit + ")")
    axis.legend()
    figure_path = save_figure(figure, name + "_forecast.png")

    return {
        "state": state, "crop": crop, "unit": unit,
        "evaluation": evaluation, "future_forecast": future_table,
        "arima_mae": arima_mae, "arima_rmse": arima_rmse,
        "naive_mae": naive_mae, "naive_rmse": naive_rmse,
        "figure": figure_path,
    }


# ---------------------------------------------------------------------------
# 8. Full report
# ---------------------------------------------------------------------------

def markdown_table(frame, maximum_rows=25):
    """Render a small DataFrame as a Markdown table without extra libraries."""
    if frame is None or frame.empty:
        return "No rows to display."
    shown = frame.head(maximum_rows)
    header = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in shown.itertuples(index=False, name=None):
        cells = ["N/A" if pd.isna(value) else str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(frame) > maximum_rows:
        lines.append("")
        lines.append("Showing " + str(maximum_rows) + " of " + str(len(frame)) +
                     " rows; see the CSV for the full table.")
    return "\n".join(lines)


def generate_report(data):
    """Run every analysis step and write outputs/KISAN_X_ANALYSIS_REPORT.md."""
    audit, missing, units = audit_data(data)
    tables = build_eda_tables(data)
    charts = make_eda_charts(data, tables)
    yield_stats, msp_stats = descriptive_statistics(data)
    correlations = rainfall_yield_correlations(data)
    shares, metrics, top_five = diversification(data)
    charts.extend(plot_diversification(metrics))

    # One rainfall-yield scatter per state, for the strongest association found
    # there. The plot illustrates the correlation; it does not prove a cause.
    if not correlations.empty:
        for state, group in correlations.groupby("State"):
            strongest = group.iloc[0]       # already sorted by |Pearson r|
            charts.append(plot_rainfall_yield(data, state, str(strongest["Crop"])))

    # One season test per state, using that state's largest-area crop.
    test_rows = []
    for state, group in tables["crop_area_by_state"].groupby("State"):
        crop = group.iloc[0]["Crop"]          # already sorted by area, descending
        try:
            test_rows.append(season_yield_test(data, state, crop))
        except ValueError as error:
            test_rows.append({"State": state, "Crop": crop,
                              "Test": "Kruskal-Wallis H-test",
                              "Conclusion": "Skipped: " + str(error)})
    hypothesis_table = pd.DataFrame(test_rows)
    save_table(hypothesis_table, "report_hypothesis_tests.csv")

    # One forecast per state, using the largest-area complete annual series.
    candidates = forecast_candidates(data)
    forecast_rows = []
    for state in sorted(data["State"].dropna().unique()):
        state_candidates = (candidates[candidates["State"] == state]
                            if not candidates.empty else candidates)
        if state_candidates.empty:
            forecast_rows.append({"State": state,
                                  "Status": "Skipped: no complete annual series."})
            continue
        selected = state_candidates.iloc[0]
        try:
            result = forecast_production(data, state, str(selected["Crop"]),
                                         str(selected["Production_Unit"]))
            forecast_rows.append({
                "State": state, "Crop": result["crop"], "Production_Unit": result["unit"],
                "Status": "Completed",
                "ARIMA_MAE": round(result["arima_mae"], 2),
                "ARIMA_RMSE": round(result["arima_rmse"], 2),
                "Naive_MAE": round(result["naive_mae"], 2),
                "Naive_RMSE": round(result["naive_rmse"], 2),
            })
            charts.append(result["figure"])
        except Exception as error:
            forecast_rows.append({"State": state, "Crop": str(selected["Crop"]),
                                  "Status": "Forecast failed: " + str(error)})
    forecast_summary = pd.DataFrame(forecast_rows)
    save_table(forecast_summary, "report_forecast_summary.csv")

    first_year = str(data.loc[data["Year_Start"].idxmin(), "Year"])
    last_year = str(data.loc[data["Year_Start"].idxmax(), "Year"])
    lines = [
        "# KISAN-X Analysis Report",
        "",
        "## 1. Dataset overview",
        "",
        "- Observations: **" + format(len(data), ",") + "**",
        "- Agricultural years: **" + first_year + " to " + last_year + "**",
        "- States: **" + ", ".join(sorted(data["State"].dropna().unique())) + "**",
        "- Production units: **" + ", ".join(sorted(data["Production_Unit"].dropna().unique())) + "**",
        "- Missing MSP values: **" + format(int(data["MSP_Rs_per_Quintal"].isna().sum()), ",") +
        "**, kept as gaps rather than imputed.",
        "",
        "## 2. Data quality checks",
        "",
        markdown_table(audit),
        "",
        "## 3. Exploratory patterns",
        "",
        "Production totals are reported separately for each production unit; Tonnes and Bales",
        "are never added together. Area figures are sums of reported crop-season records, so",
        "they are not a measure of unique physical farmland.",
        "",
        "### Leading crops by cumulative reported area",
        "",
    ]
    for state, group in tables["top_crops"].groupby("State"):
        lines.append("**" + str(state) + ":** " +
                     ", ".join(str(crop) for crop in group["Crop"].head(3)))
        lines.append("")

    lines += [
        "## 4. Rainfall and yield relationships",
        "",
        "Rainfall is a state-level IMD proxy repeated on district rows, so records are first",
        "aggregated to one State-Year-Crop-Unit observation. Correlation is association only.",
        "Many combinations are tested, so the strongest results are exploratory.",
        "",
        "### Largest absolute Pearson associations",
        "",
    ]
    if correlations.empty:
        lines.append("No series met the minimum observation requirements.")
    else:
        for row in correlations.head(5).itertuples():
            lines.append("- " + row.State + ", " + row.Crop + " (" + row.Production_Unit + "), " +
                         row.Rainfall_Measure + ": r = " + format(row.Pearson_r, ".3f") +
                         ", p = " + format(row.Pearson_p_value, ".4f") +
                         " (n = " + str(row.Sample_Years) + " years).")

    lines += [
        "",
        "## 5. Crop diversification",
        "",
        "- **HHI:** sum of squared crop area shares. Higher means area concentrated in fewer crops.",
        "- **Diversity indicator:** 1 - HHI.",
        "- **Effective number of crops:** 1 / HHI.",
        "",
        "These describe reported area shares only, not profitability or suitability.",
        "",
        "## 6. Seasonal yield hypothesis tests",
        "",
        markdown_table(hypothesis_table[
            [c for c in ["State", "Crop", "H_Statistic", "P_Value", "Conclusion"]
             if c in hypothesis_table.columns]
        ]),
        "",
        "## 7. Forecasting",
        "",
        "For each state the largest-area crop with at least " + str(MIN_FORECAST_YEARS) +
        " consecutive annual observations",
        "was modelled with ARIMA(1,1,0) and compared against a naive previous-year baseline on",
        "a chronological holdout. Lower MAE and RMSE are better.",
        "",
        markdown_table(forecast_summary),
        "",
        "Forecasts use past production only. Future rainfall and MSP are not inputs and the",
        "short annual history limits confidence; these are estimates, not guarantees.",
        "",
        "## 8. Output files",
        "",
        "- `outputs/tables/` - audit, descriptive statistics, trends, correlations, shares, tests.",
        "- `outputs/figures/` - " + str(len(charts)) + " labelled Matplotlib/Seaborn charts.",
        "- `outputs/forecasts/` - holdout evaluations and future-year predictions.",
        "",
        "## 9. Main limitations",
        "",
        "1. Rainfall is a state-level proxy, not district rainfall.",
        "2. MSP covers only matched crop-years and is a price, not profit.",
        "3. Production units differ and must stay separated in every total.",
        "4. Area sums are reported crop-season records, not unique farmland.",
        "5. Correlation is not causation; input costs, irrigation, soil and demand are absent.",
        "6. Forecast accuracy on one short holdout period is uncertain.",
    ]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    report_path = os.path.join(OUTPUT_ROOT, "KISAN_X_ANALYSIS_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")

    return {"report_path": report_path, "forecast_summary": forecast_summary,
            "charts": charts, "audit": audit}
