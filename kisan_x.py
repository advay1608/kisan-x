"""KISAN-X analysis module.

Every analysis function lives here. main.py only shows the menu and calls them.
The sections below follow the same order as the menu (1 to 8).

Rules that every function follows:
  * The source CSV is only read, never changed.
  * Production is never added across different units (Tonnes vs Bales).
  * Summed area is "reported crop-season area", not unique farmland.
  * Rainfall is one state-level value copied onto every district row.
  * Correlation shows association only, never cause and effect.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid", context="notebook")


# ===========================================================================
# Settings
# ===========================================================================

REQUIRED_COLUMNS = [
    "Year", "State", "District", "Crop", "Season", "Area_Ha",
    "Production", "Production_Unit", "Yield", "Annual_Rainfall_mm",
    "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal",
]
NUMBER_COLUMNS = ["Area_Ha", "Production", "Yield", "Annual_Rainfall_mm",
                  "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal"]
TEXT_COLUMNS = ["Year", "State", "District", "Crop", "Season", "Production_Unit"]

# One row should exist for each combination of these columns.
KEY_COLUMNS = ["Year", "State", "District", "Crop", "Season"]

# The two rainfall columns and the label used for each in the results.
RAINFALL_MEASURES = {
    "Annual_Rainfall_mm": "Annual rainfall",
    "JJAS_Rainfall_mm": "Monsoon (JJAS) rainfall",
}

# Folder paths (EX 5: os module). Paths are built from this file's location,
# so the program works no matter which folder it is started from.
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_FOLDER, "data", "this_file.csv")
OUTPUT_ROOT = os.path.join(PROJECT_FOLDER, "outputs")

MIN_CORRELATION_YEARS = 8    # fewest years needed to calculate a correlation
MIN_SEASON_VALUES = 5        # fewest yield values a season needs for the test
MIN_FORECAST_YEARS = 15      # fewest consecutive years needed to forecast
HOLDOUT_YEARS = 3            # last years kept aside to test the forecast
FUTURE_YEARS = 3             # years forecast into the future


# ===========================================================================
# Small helpers for saving files
# ===========================================================================

def make_output_directories():
    """Create outputs/tables, outputs/figures and outputs/forecasts if missing."""
    for folder in ["tables", "figures", "forecasts"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)


def save_table(table, filename, folder="tables"):
    """Save a DataFrame as a CSV file in outputs/<folder>/ and return its path."""
    path = os.path.join(OUTPUT_ROOT, folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return path


def save_figure(figure, filename):
    """Save a chart as a PNG file in outputs/figures/ and return its path."""
    path = os.path.join(OUTPUT_ROOT, "figures", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)            # free the memory used by the chart
    return path


def safe_name(*parts):
    """Join labels into a file-safe name, e.g. ("Uttar Pradesh", "Rice") -> Uttar_Pradesh_Rice."""
    text = "_".join(str(part) for part in parts)
    return text.replace("/", "-").replace(" ", "_")


def line_chart_by_state(table, y_column, title, y_label, filename):
    """Draw one line per state over the years and save the chart."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=table, x="Year_Start", y=y_column, hue="State", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Agricultural year (starting year)")
    axis.set_ylabel(y_label)
    return save_figure(figure, filename)


# ===========================================================================
# Loading the data
# ===========================================================================

def load_data(file_path=DATA_PATH):
    """Read the CSV, check the 12 required columns and prepare column types.

    Values that cannot be read as numbers become NaN (missing) so the audit can
    count them. No rows are deleted and nothing is written back to the file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}\n"
                                "Place the final CSV at data/this_file.csv.")

    data = pd.read_csv(file_path, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError("The dataset is missing required columns: " + ", ".join(missing_columns) +
                         "\nExpected columns: " + ", ".join(REQUIRED_COLUMNS))

    for column in NUMBER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    # "1997-98" -> 1997, so years can be sorted and used in time series.
    data["Year_Start"] = pd.to_numeric(data["Year"].str[:4], errors="coerce")
    return data


def annual_series(data):
    """Combine district and season rows into one row per Year + State + Crop + Unit.

    * Area and production are added up (only inside one unit, so this is safe).
    * Rainfall is the same for every district in a state-year, so we keep it
      once ("first") instead of counting each district copy as a new value.
    * Weighted yield = total production / total area.
    """
    annual = data.groupby(["Year_Start", "Year", "State", "Crop", "Production_Unit"],
                          as_index=False).agg(
        Area_Ha=("Area_Ha", "sum"),
        Production=("Production", "sum"),
        Annual_Rainfall_mm=("Annual_Rainfall_mm", "first"),
        JJAS_Rainfall_mm=("JJAS_Rainfall_mm", "first"),
    )
    # Zero area would mean dividing by zero, so it is treated as missing.
    annual["Weighted_Yield"] = annual["Production"] / annual["Area_Ha"].replace(0, np.nan)
    return annual.sort_values(["State", "Crop", "Production_Unit", "Year_Start"])


# ===========================================================================
# 1. Data audit (EX 8: missing values, duplicates, checks)
# ===========================================================================

def audit_data(data):
    """Check the data quality and save three summary tables.

    Missing MSP values are expected (not every crop has an MSP), so they are
    counted but never filled with zero or the mean. Duplicates are counted,
    never deleted.
    """
    # Yield should equal Production / Area. A tiny tolerance allows for decimals.
    calculated_yield = data["Production"] / data["Area_Ha"].replace(0, np.nan)
    yield_matches = np.isclose(calculated_yield, data["Yield"],
                               rtol=1e-7, atol=1e-7, equal_nan=True)
    yield_mismatches = int((~yield_matches).sum())

    # Every State + Year should have exactly one rainfall value.
    rainfall_values = data.groupby(["State", "Year_Start"])[
        ["Annual_Rainfall_mm", "JJAS_Rainfall_mm"]].nunique()
    rainfall_conflicts = int((rainfall_values > 1).any(axis=1).sum())

    first_year = data.loc[data["Year_Start"].idxmin(), "Year"]
    last_year = data.loc[data["Year_Start"].idxmax(), "Year"]

    rows = [
        ("Rows", len(data), "Observations loaded"),
        ("Columns", len(data.columns) - 1, "Required columns (Year_Start is derived in memory)"),
        ("Exact duplicate rows", int(data.duplicated().sum()), "Expected 0; counted, not deleted"),
        ("Duplicate analytical keys", int(data.duplicated(KEY_COLUMNS).sum()),
         "Key: Year + State + District + Crop + Season; expected 0"),
        ("States", data["State"].nunique(), "Unique states represented"),
        ("Districts", data["District"].nunique(), "Unique district labels"),
        ("Crops", data["Crop"].nunique(), "Unique crop labels"),
        ("Seasons", data["Season"].nunique(), "Unique crop-season labels"),
        ("Production units", data["Production_Unit"].nunique(),
         "Totals are only ever formed within one unit"),
        ("First agricultural year", str(first_year), "Earliest year label"),
        ("Last agricultural year", str(last_year), "Latest year label"),
        ("Missing MSP values", int(data["MSP_Rs_per_Quintal"].isna().sum()),
         "Expected coverage gap; not replaced with zero or the mean"),
        ("Negative area values", int((data["Area_Ha"] < 0).sum()), "Expected 0"),
        ("Negative production values", int((data["Production"] < 0).sum()), "Expected 0"),
        ("Negative yield values", int((data["Yield"] < 0).sum()), "Expected 0"),
        ("Yield inconsistent with Production / Area", yield_mismatches,
         "Tolerance 1e-7; investigate if non-zero"),
        ("State-year rainfall conflicts", rainfall_conflicts,
         "Expected 0: the proxy should hold one value per State + Year"),
    ]
    audit = pd.DataFrame(rows, columns=["Metric", "Value", "Interpretation"])
    save_table(audit, "data_audit_summary.csv")

    # Missing values in each column.
    missing = pd.DataFrame({"Column": data.columns, "Missing_Count": data.isna().sum().values})
    missing["Missing_Percent"] = (missing["Missing_Count"] / len(data) * 100).round(4)
    missing["Note"] = np.where(missing["Column"] == "MSP_Rs_per_Quintal",
                               "Expected coverage gap", "Review if non-zero")
    missing = missing.sort_values("Missing_Count", ascending=False)
    save_table(missing, "missing_values_by_column.csv")

    # How many records use each production unit.
    units = data.groupby("Production_Unit", dropna=False, as_index=False).agg(
        Records=("Production", "size"),
        Distinct_Crops=("Crop", "nunique"),
        Total_Reported_Area_Ha=("Area_Ha", "sum"),
    )
    save_table(units, "production_unit_coverage.csv")
    return audit, missing, units


# ===========================================================================
# 2. Exploratory analysis (EX 8 grouping, EX 9-10 charts)
# ===========================================================================

def build_eda_tables(data):
    """Build and save the area, production and top-crop summary tables."""
    # Area added across district records. The same field can be counted under
    # several crops or seasons, so this is NOT unique land area.
    area_by_state_year = data.groupby(["Year_Start", "Year", "State"], as_index=False).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"), Records=("Area_Ha", "size"))
    save_table(area_by_state_year, "area_by_state_year.csv")

    # Production is grouped by unit too, so Tonnes and Bales are never added.
    production_by_state_year = data.groupby(
        ["Year_Start", "Year", "State", "Production_Unit"], as_index=False).agg(
        Production_Total=("Production", "sum"), Records=("Production", "size"))
    save_table(production_by_state_year, "production_by_state_year_unit.csv")

    # Total reported area of each crop in each state, largest first.
    crop_area_by_state = data.groupby(["State", "Crop"], as_index=False).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"), Records=("Area_Ha", "size"))
    crop_area_by_state = crop_area_by_state.sort_values(
        ["State", "Reported_Crop_Area_Ha"], ascending=[True, False])
    save_table(crop_area_by_state, "crop_area_by_state.csv")

    # Top 10 crops of each state, numbered 1 to 10.
    top_crops = crop_area_by_state.groupby("State").head(10).copy()
    top_crops["Rank_By_Reported_Area"] = top_crops.groupby("State").cumcount() + 1
    save_table(top_crops, "top_crops_by_state_area.csv")

    return {
        "area_by_state_year": area_by_state_year,
        "production_by_state_year": production_by_state_year,
        "crop_area_by_state": crop_area_by_state,
        "top_crops": top_crops,
    }


def make_eda_charts(data, tables):
    """Save the line, bar and histogram charts. Returns the list of file paths."""
    paths = []

    # Line chart: reported area over time.
    paths.append(line_chart_by_state(
        tables["area_by_state_year"], "Reported_Crop_Area_Ha",
        "Reported Crop Area by State and Agricultural Year",
        "Sum of reported crop-season area (ha)",
        "reported_crop_area_trends_by_state.png"))

    # Line charts: production over time, one chart per unit.
    for unit, group in tables["production_by_state_year"].groupby("Production_Unit"):
        paths.append(line_chart_by_state(
            group, "Production_Total",
            f"Reported Production Trend by State (unit: {unit})",
            f"Production ({unit})",
            f"production_trends_{safe_name(unit)}.png"))

    # Bar charts: top crops by area, one chart per state.
    for state, group in tables["top_crops"].groupby("State"):
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.barplot(data=group.sort_values("Reported_Crop_Area_Ha"),
                    x="Reported_Crop_Area_Ha", y="Crop", color="steelblue", ax=axis)
        axis.set_title(f"Top Crops by Reported Area - {state}")
        axis.set_xlabel("Sum of reported crop-season area (ha)")
        axis.set_ylabel("Crop")
        paths.append(save_figure(figure, f"top_crops_area_{safe_name(state)}.png"))

    # Histogram: yield values for crops measured in tonnes.
    tonne_yield = data.loc[data["Production_Unit"] == "Tonnes", "Yield"].dropna()
    if len(tonne_yield) > 0:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        sns.histplot(tonne_yield, bins=40, kde=True, color="steelblue", ax=axis)
        axis.set_title("Yield Distribution for Records Measured in Tonnes")
        axis.set_xlabel("Yield (tonnes per hectare)")
        axis.set_ylabel("Record count")
        paths.append(save_figure(figure, "yield_distribution_tonnes.png"))

    return paths


# ===========================================================================
# 3. Descriptive statistics (EX 7 NumPy, EX 11 SciPy)
# ===========================================================================

def descriptive_statistics(data):
    """Describe yield for each State + Crop + Unit, and MSP for each crop.

    Units stay separate so tonnes/ha and bales/ha are never mixed in one mean.
    """
    rows = []
    for (state, crop, unit), group in data.groupby(["State", "Crop", "Production_Unit"]):
        values = group["Yield"].dropna().to_numpy(dtype=float)
        count = len(values)
        if count == 0:
            continue

        row = {
            "State": state, "Crop": crop, "Production_Unit": unit,
            "Count": count,
            "Mean_Yield": np.mean(values),
            "Median_Yield": np.median(values),
            "Standard_Deviation": np.nan,
            "Variance": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Minimum": np.min(values),
            "Maximum": np.max(values),
        }
        # Spread needs at least 2 values, skewness 3 and kurtosis 4.
        # ddof=1 and bias=False give the usual "sample" versions of each measure.
        if count >= 2:
            row["Standard_Deviation"] = np.std(values, ddof=1)
            row["Variance"] = np.var(values, ddof=1)
        if count >= 3:
            row["Skewness"] = stats.skew(values, bias=False)
        if count >= 4:
            row["Kurtosis"] = stats.kurtosis(values, bias=False)
        rows.append(row)

    yield_stats = pd.DataFrame(rows).sort_values(["State", "Crop", "Production_Unit"])
    save_table(yield_stats, "descriptive_yield_statistics.csv")

    # MSP is one price per crop per year, copied onto every district row.
    # We keep each crop-year price once, for this summary only.
    msp = data[["Year", "Crop", "MSP_Rs_per_Quintal"]].dropna().drop_duplicates()
    msp_stats = msp.groupby("Crop", as_index=False).agg(
        Distinct_Crop_Years=("Year", "nunique"),
        Mean_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "mean"),
        Minimum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "min"),
        Maximum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "max"),
    )
    save_table(msp_stats, "descriptive_msp_statistics.csv")
    return yield_stats, msp_stats


# ===========================================================================
# 4. Rainfall and yield (EX 11 correlation)
# ===========================================================================

def rainfall_yield_correlations(data):
    """Correlate rainfall with weighted yield for every State + Crop + Unit.

    annual_series() runs first so each state-year rainfall value is used once.
    Without it, every district row would repeat the same rainfall number and
    make the sample look bigger than it really is.
    """
    annual = annual_series(data)
    rows = []
    for (state, crop, unit), group in annual.groupby(["State", "Crop", "Production_Unit"]):
        for column, label in RAINFALL_MEASURES.items():
            # Use only years where both rainfall and yield are present.
            pair = group[[column, "Weighted_Yield"]].dropna()
            if len(pair) < MIN_CORRELATION_YEARS:
                continue
            rainfall = pair[column].to_numpy(dtype=float)
            yields = pair["Weighted_Yield"].to_numpy(dtype=float)
            if np.std(rainfall) == 0 or np.std(yields) == 0:
                continue         # a flat line has no correlation to measure

            pearson = stats.pearsonr(rainfall, yields)     # straight-line relationship
            spearman = stats.spearmanr(rainfall, yields)   # rank (up/down together) relationship
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
        # Strongest relationships (positive or negative) first.
        results["Absolute_Pearson_r"] = results["Pearson_r"].abs()
        results = results.sort_values("Absolute_Pearson_r", ascending=False)
    save_table(results, "rainfall_yield_correlations.csv")
    return results


def plot_rainfall_yield(data, state, crop):
    """Scatter plot of monsoon rainfall against weighted yield for one state and crop."""
    annual = annual_series(data)
    subset = annual[(annual["State"] == state) & (annual["Crop"] == crop)]
    if subset.empty:
        raise ValueError(f"No annual records found for {state} / {crop}.")

    figure, axis = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=subset, x="JJAS_Rainfall_mm", y="Weighted_Yield",
                    hue="Production_Unit", s=75, ax=axis)
    axis.set_title(f"Monsoon Rainfall and Crop Yield - {state} / {crop}")
    axis.set_xlabel("JJAS rainfall proxy (mm)")
    axis.set_ylabel("Weighted yield (production unit per hectare)")
    return save_figure(figure, f"rainfall_yield_{safe_name(state, crop)}.png")


# ===========================================================================
# 5. Season hypothesis test (EX 11 Kruskal-Wallis)
# ===========================================================================

def season_yield_test(data, state, crop, alpha=0.05):
    """Test whether yield differs between seasons for one state and crop.

    Null hypothesis (H0): yield is distributed the same way in every season.
    Kruskal-Wallis compares ranks, so it does not need normally distributed
    data. Rows share districts and years, so the result is exploratory only.
    """
    subset = data[(data["State"] == state) & (data["Crop"] == crop)]

    # Keep only seasons with enough yield values to compare.
    season_values = {}
    for season, group in subset.groupby("Season"):
        values = group["Yield"].dropna().to_numpy(dtype=float)
        if len(values) >= MIN_SEASON_VALUES:
            season_values[str(season)] = values

    if len(season_values) < 2:
        raise ValueError("Not enough data for a season comparison. Choose a crop with at least "
                         f"two seasons that each have {MIN_SEASON_VALUES} or more yield values.")

    result = stats.kruskal(*season_values.values())
    p_value = float(result.pvalue)
    reject_null = p_value < alpha

    if reject_null:
        conclusion = "Evidence of a difference at the chosen significance level."
    else:
        conclusion = "Insufficient evidence of a difference at the chosen significance level."

    counts = [f"{season}: {len(values)}" for season, values in season_values.items()]
    summary = {
        "State": state,
        "Crop": crop,
        "Test": "Kruskal-Wallis H-test",
        "Seasons_Compared": ", ".join(season_values),
        "Observations_Per_Season": "; ".join(counts),
        "H_Statistic": float(result.statistic),
        "P_Value": p_value,
        "Alpha": alpha,
        "Reject_Null": bool(reject_null),
        "Null_Hypothesis": "Yield distributions are the same across the selected seasons.",
        "Conclusion": conclusion,
        "Caveat": "Exploratory; records share districts and years, so no causal claim.",
    }
    save_table(pd.DataFrame([summary]), "latest_season_hypothesis_test.csv")
    return summary


# ===========================================================================
# 6. Crop diversification (EX 7 vectorised arithmetic)
# ===========================================================================

def diversification(data):
    """Work out each crop's share of area and how concentrated each state is.

    share        = crop area / total area of the state in that year
    HHI          = sum of squared shares (0 to 1; higher = fewer crops dominate)
    1 - HHI      = diversity indicator (higher = more even mix of crops)
    1 / HHI      = effective number of equally sized crops
    These describe area only, not profit or which crop suits the land.
    """
    crop_area = data.groupby(["Year_Start", "Year", "State", "Crop"], as_index=False).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"))

    # transform("sum") puts the state-year total on every crop row.
    state_total = crop_area.groupby(["Year_Start", "State"])["Reported_Crop_Area_Ha"].transform("sum")
    crop_area["Area_Share"] = crop_area["Reported_Crop_Area_Ha"] / state_total
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

    # The five crops with the largest share in each state and year.
    top_five = crop_area.sort_values(["State", "Year_Start", "Area_Share_Percent"],
                                     ascending=[True, True, False])
    top_five = top_five.groupby(["Year_Start", "State"]).head(5)
    save_table(top_five, "top_five_crop_shares_by_year.csv")
    return crop_area, metrics, top_five


def plot_diversification(metrics):
    """Save line charts of the diversity indicator and HHI for each state."""
    diversity_chart = line_chart_by_state(
        metrics, "Area_Diversity_Index", "Crop-Area Diversity by State and Year",
        "1 - HHI (higher means more even shares)", "area_diversity_by_state.png")
    concentration_chart = line_chart_by_state(
        metrics, "HHI_Area_Concentration", "Crop-Area Concentration by State and Year",
        "HHI (higher means more concentrated)", "area_concentration_by_state.png")
    return [diversity_chart, concentration_chart]


# ===========================================================================
# 7. Forecasting (stationarity test, ARIMA, holdout evaluation)
# ===========================================================================

def has_consecutive_years(years):
    """True if there are enough years and none are missing (e.g. 1997, 1998, 1999...)."""
    years = np.sort(np.asarray(years, dtype=int))
    return len(years) >= MIN_FORECAST_YEARS and np.all(np.diff(years) == 1)


def forecast_candidates(data):
    """List the State + Crop + Unit series that are long enough to forecast.

    Series with a missing year are left out. Filling the gap with zero would
    invent a crop failure that never happened.
    """
    annual = annual_series(data)
    rows = []
    for (state, crop, unit), group in annual.groupby(["State", "Crop", "Production_Unit"]):
        years = group["Year_Start"]
        if not has_consecutive_years(years):
            continue
        rows.append({
            "State": state, "Crop": crop, "Production_Unit": unit,
            "Annual_Points": len(years),
            "First_Year": int(years.min()), "Last_Year": int(years.max()),
            "Average_Annual_Reported_Area_Ha": float(group["Area_Ha"].mean()),
        })

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["State", "Average_Annual_Reported_Area_Ha"],
                                            ascending=[True, False])
    return candidates


def production_series(data, state, crop, unit):
    """Return yearly production for one State + Crop + Unit, after checking it is usable."""
    annual = annual_series(data)
    series = annual[(annual["State"] == state) & (annual["Crop"] == crop) &
                    (annual["Production_Unit"] == unit)]
    if series.empty:
        raise ValueError("No records found for the selected state, crop and unit.")
    if len(series) < MIN_FORECAST_YEARS:
        raise ValueError(f"At least {MIN_FORECAST_YEARS} annual points are required.")
    if not has_consecutive_years(series["Year_Start"]):
        raise ValueError("This series has missing years. KISAN-X does not fill gaps with zero; "
                         "choose a complete series instead.")
    if series["Production"].nunique() < 3:
        raise ValueError("This series has too little variation to forecast usefully.")
    return series[["Year_Start", "Year", "Production"]]


def year_label(year_start):
    """Turn 1997 into the agricultural-year label "1997-98"."""
    year_start = int(year_start)
    return f"{year_start}-{(year_start + 1) % 100:02d}"


def error_metrics(actual, predicted):
    """Return MAE (average error size) and RMSE (punishes big errors more)."""
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(mae), float(rmse)


def stationarity_test(values):
    """Augmented Dickey-Fuller test. Returns (statistic, p-value, note).

    A large p-value means the series has a trend (not stationary), which is
    why the ARIMA model differences the data once (the middle 1 in (1,1,0)).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # hide technical statsmodels messages
            result = adfuller(values, autolag="AIC")
    except Exception as error:
        return np.nan, np.nan, f"Stationarity test unavailable: {error}"

    statistic, p_value = float(result[0]), float(result[1])
    if p_value < 0.05:
        note = "Evidence against a unit root at 5%."
    else:
        note = "No evidence against a unit root at 5%; differencing is justified."
    return statistic, p_value, note


def fit_arima(values):
    """Fit the fixed ARIMA(1,1,0) model.

    (1,1,0) = 1 autoregressive term, data differenced 1 time, 0 moving-average
    terms. The order is fixed on purpose instead of searched automatically.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # hide technical statsmodels messages
        model = ARIMA(values, order=(1, 1, 0),
                      enforce_stationarity=False, enforce_invertibility=False)
        return model.fit()


def forecast_production(data, state, crop, unit):
    """Test ARIMA(1,1,0) against a simple baseline, then forecast future years.

    Step 1: split the years in time order - train on early years, test on the last 3.
    Step 2: run the stationarity test on the training years.
    Step 3: baseline prediction = "same as last year".
    Step 4: ARIMA trained on the early years predicts the last 3 years.
    Step 5: refit ARIMA on all years and forecast the next 3 years.
    Step 6: save the tables and the chart.
    """
    series = production_series(data, state, crop, unit)
    years = series["Year_Start"].to_numpy(dtype=int)
    values = series["Production"].to_numpy(dtype=float)

    # Step 1: time-ordered split (never shuffled, or the model would see the future).
    train = values[:-HOLDOUT_YEARS]
    test = values[-HOLDOUT_YEARS:]
    test_labels = series["Year"].astype(str).tolist()[-HOLDOUT_YEARS:]

    # Step 2: stationarity test.
    adf_statistic, adf_p_value, adf_note = stationarity_test(train)

    # Step 3: naive baseline - each test year is predicted by the year before it.
    naive_predictions = values[-HOLDOUT_YEARS - 1:-1]
    naive_mae, naive_rmse = error_metrics(test, naive_predictions)

    # Step 4: ARIMA on the training years only.
    arima_predictions = fit_arima(train).forecast(steps=HOLDOUT_YEARS)
    arima_mae, arima_rmse = error_metrics(test, arima_predictions)

    # Step 5: ARIMA on all years, forecasting the future with a 95% range.
    future = fit_arima(values).get_forecast(steps=FUTURE_YEARS)
    future_values = np.asarray(future.predicted_mean, dtype=float)
    interval = np.asarray(future.conf_int(alpha=0.05), dtype=float)
    future_years = np.arange(years[-1] + 1, years[-1] + 1 + FUTURE_YEARS)

    # Step 6: save results.
    future_table = pd.DataFrame({
        "State": state, "Crop": crop, "Production_Unit": unit,
        "Forecast_Year": [year_label(year) for year in future_years],
        # Production cannot be negative, so a floored copy is shown beside
        # the raw model output. Keeping both makes the correction visible.
        "ARIMA_Raw_Forecast": future_values,
        "Forecast_Production_NonNegative": np.maximum(future_values, 0.0),
        "Lower_95_Percent": interval[:, 0],
        "Upper_95_Percent": interval[:, 1],
        "Model": "ARIMA(1,1,0)",
        "Note": "Estimate from past production only; rainfall and MSP are not inputs.",
    })

    evaluation = pd.DataFrame({
        "State": state, "Crop": crop, "Production_Unit": unit,
        "Model": ["Naive previous-year baseline", "ARIMA(1,1,0)"],
        "Training_Years": len(train), "Holdout_Years": len(test),
        "Holdout_First_Year": test_labels[0], "Holdout_Last_Year": test_labels[-1],
        "MAE": [naive_mae, arima_mae],
        "RMSE": [naive_rmse, arima_rmse],
        "ADF_Statistic_Training": adf_statistic,
        "ADF_P_Value_Training": adf_p_value,
        "Stationarity_Note": adf_note,
    })

    name = safe_name(state, crop, unit)
    save_table(future_table, f"{name}_future_forecast.csv", folder="forecasts")
    save_table(evaluation, f"{name}_evaluation.csv", folder="forecasts")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(years[:-HOLDOUT_YEARS], train, marker="o", label="Training observations")
    axis.plot(years[-HOLDOUT_YEARS:], test, marker="o", linestyle="--", label="Actual holdout")
    axis.plot(years[-HOLDOUT_YEARS:], arima_predictions, marker="x", linestyle="--",
              label="ARIMA holdout forecast")
    axis.plot(future_years, future_values, marker="o", linestyle=":", label="Future ARIMA forecast")
    axis.set_title(f"Crop Production Forecast - {state} / {crop}")
    axis.set_xlabel("Agricultural year (starting year)")
    axis.set_ylabel(f"Production ({unit})")
    axis.legend()
    figure_path = save_figure(figure, f"{name}_forecast.png")

    return {
        "state": state, "crop": crop, "unit": unit,
        "evaluation": evaluation, "future_forecast": future_table,
        "arima_mae": arima_mae, "arima_rmse": arima_rmse,
        "naive_mae": naive_mae, "naive_rmse": naive_rmse,
        "figure": figure_path,
    }


# ===========================================================================
# 8. Full report
# ===========================================================================

def markdown_table(table, max_rows=25):
    """Turn a small DataFrame into a Markdown table (text with | separators)."""
    if table is None or table.empty:
        return "No rows to display."

    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.head(max_rows).itertuples(index=False):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("N/A")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")

    if len(table) > max_rows:
        lines.append("")
        lines.append(f"Showing {max_rows} of {len(table)} rows; see the CSV for the full table.")
    return "\n".join(lines)


def report_rainfall_plots(data, correlations):
    """For each state, plot the crop with the strongest rainfall-yield correlation."""
    paths = []
    if correlations.empty:
        return paths
    for state, group in correlations.groupby("State"):
        strongest_crop = str(group.iloc[0]["Crop"])     # table is already sorted by |r|
        paths.append(plot_rainfall_yield(data, state, strongest_crop))
    return paths


def report_season_tests(data, crop_area_by_state):
    """Run the season test on each state's largest-area crop and save a summary table."""
    rows = []
    for state, group in crop_area_by_state.groupby("State"):
        crop = group.iloc[0]["Crop"]                    # table is already sorted by area
        try:
            rows.append(season_yield_test(data, state, crop))
        except ValueError as error:
            rows.append({"State": state, "Crop": crop, "Test": "Kruskal-Wallis H-test",
                         "Conclusion": f"Skipped: {error}"})
    hypothesis_table = pd.DataFrame(rows)
    save_table(hypothesis_table, "report_hypothesis_tests.csv")
    return hypothesis_table


def report_forecasts(data):
    """Forecast each state's largest-area complete series. Returns (summary table, chart paths)."""
    candidates = forecast_candidates(data)
    rows = []
    charts = []
    for state in sorted(data["State"].dropna().unique()):
        if candidates.empty or state not in candidates["State"].values:
            rows.append({"State": state, "Status": "Skipped: no complete annual series."})
            continue

        best = candidates[candidates["State"] == state].iloc[0]   # sorted by area
        crop, unit = str(best["Crop"]), str(best["Production_Unit"])
        try:
            result = forecast_production(data, state, crop, unit)
        except Exception as error:
            rows.append({"State": state, "Crop": crop, "Status": f"Forecast failed: {error}"})
            continue

        rows.append({
            "State": state, "Crop": crop, "Production_Unit": unit,
            "Status": "Completed",
            "ARIMA_MAE": round(result["arima_mae"], 2),
            "ARIMA_RMSE": round(result["arima_rmse"], 2),
            "Naive_MAE": round(result["naive_mae"], 2),
            "Naive_RMSE": round(result["naive_rmse"], 2),
        })
        charts.append(result["figure"])

    forecast_summary = pd.DataFrame(rows)
    save_table(forecast_summary, "report_forecast_summary.csv")
    return forecast_summary, charts


def generate_report(data):
    """Run every analysis step and write outputs/KISAN_X_ANALYSIS_REPORT.md."""
    # Step 1: run all analyses, so the report and the saved files always match.
    audit, missing, units = audit_data(data)
    tables = build_eda_tables(data)
    charts = make_eda_charts(data, tables)
    descriptive_statistics(data)
    correlations = rainfall_yield_correlations(data)
    shares, metrics, top_five = diversification(data)
    charts += plot_diversification(metrics)
    charts += report_rainfall_plots(data, correlations)
    hypothesis_table = report_season_tests(data, tables["crop_area_by_state"])
    forecast_summary, forecast_charts = report_forecasts(data)
    charts += forecast_charts

    # Step 2: write the results as Markdown text.
    report_path = write_report(data, audit, tables["top_crops"], correlations,
                               hypothesis_table, forecast_summary, len(charts))
    return {"report_path": report_path, "forecast_summary": forecast_summary,
            "charts": charts, "audit": audit}


def write_report(data, audit, top_crops, correlations, hypothesis_table,
                 forecast_summary, chart_count):
    """Write the Markdown report file and return its path."""
    first_year = data.loc[data["Year_Start"].idxmin(), "Year"]
    last_year = data.loc[data["Year_Start"].idxmax(), "Year"]
    states = ", ".join(sorted(data["State"].dropna().unique()))
    units = ", ".join(sorted(data["Production_Unit"].dropna().unique()))
    missing_msp = int(data["MSP_Rs_per_Quintal"].isna().sum())

    lines = [
        "# KISAN-X Analysis Report",
        "",
        "## 1. Dataset overview",
        "",
        f"- Observations: **{len(data):,}**",
        f"- Agricultural years: **{first_year} to {last_year}**",
        f"- States: **{states}**",
        f"- Production units: **{units}**",
        f"- Missing MSP values: **{missing_msp:,}**, kept as gaps rather than imputed.",
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
    for state, group in top_crops.groupby("State"):
        lines.append(f"**{state}:** " + ", ".join(group["Crop"].head(3).astype(str)))
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
            lines.append(f"- {row.State}, {row.Crop} ({row.Production_Unit}), "
                         f"{row.Rainfall_Measure}: r = {row.Pearson_r:.3f}, "
                         f"p = {row.Pearson_p_value:.4f} (n = {row.Sample_Years} years).")

    # The season-test table may lack some columns if every test was skipped.
    test_columns = [column for column in ["State", "Crop", "H_Statistic", "P_Value", "Conclusion"]
                    if column in hypothesis_table.columns]

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
        markdown_table(hypothesis_table[test_columns]),
        "",
        "## 7. Forecasting",
        "",
        f"For each state the largest-area crop with at least {MIN_FORECAST_YEARS} "
        "consecutive annual observations",
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
        f"- `outputs/figures/` - {chart_count} labelled Matplotlib/Seaborn charts.",
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
    return report_path
