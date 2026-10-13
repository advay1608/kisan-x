"""Descriptive statistics, rainfall correlations and hypothesis tests."""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from data_utils import add_year_start


def run_descriptive_statistics(data, output_root):
    """Save yield statistics by comparable crop/unit groups and unique MSP summaries."""
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    yield_rows = []
    for keys, group in data.groupby(["State", "Crop", "Production_Unit"], dropna=False):
        state, crop, unit = keys
        values = pd.to_numeric(group["Yield"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        if len(values) >= 2:
            description = stats.describe(values, ddof=1, bias=False)
            variance = float(description.variance)
            skewness = float(description.skewness) if len(values) >= 3 else np.nan
            kurtosis = float(description.kurtosis) if len(values) >= 4 else np.nan
        else:
            variance, skewness, kurtosis = np.nan, np.nan, np.nan
        yield_rows.append({
            "State": state,
            "Crop": crop,
            "Production_Unit": unit,
            "Count": len(values),
            "Mean_Yield": float(np.mean(values)),
            "Median_Yield": float(np.median(values)),
            "Standard_Deviation": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
            "Variance_SciPy": variance,
            "Skewness_SciPy": skewness,
            "Kurtosis_SciPy": kurtosis,
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
        })
    yield_stats = pd.DataFrame(yield_rows).sort_values(["State", "Crop", "Production_Unit"])
    yield_stats.to_csv(os.path.join(tables_dir, "descriptive_yield_statistics.csv"), index=False)

    # MSP appears repeatedly on district records after joining. Deduplicate only
    # for this MSP summary calculation; the original data are never changed.
    msp = data[["Year", "Crop", "MSP_Rs_per_Quintal"]].dropna().drop_duplicates()
    msp_stats = msp.groupby("Crop", as_index=False).agg(
        Distinct_Crop_Years=("Year", "nunique"),
        Mean_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "mean"),
        Minimum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "min"),
        Maximum_MSP_Rs_per_Quintal=("MSP_Rs_per_Quintal", "max"),
    ).sort_values("Crop")
    msp_stats.to_csv(os.path.join(tables_dir, "descriptive_msp_statistics.csv"), index=False)
    return yield_stats, msp_stats


def build_state_year_crop_yield(data):
    """Aggregate district records to one row per State-Year-Crop-Unit.

    Rainfall is a state-level proxy repeated in district rows. The aggregation
    prevents those repeated copies from being treated as independent rainfall
    observations in correlations.
    """
    work = add_year_start(data)
    grouped = work.groupby(
        ["Year_Start", "Year", "State", "Crop", "Production_Unit"], as_index=False
    ).agg(
        Area_Ha=("Area_Ha", "sum"),
        Production=("Production", "sum"),
        Annual_Rainfall_mm=("Annual_Rainfall_mm", "first"),
        JJAS_Rainfall_mm=("JJAS_Rainfall_mm", "first"),
    )
    grouped["Weighted_Yield"] = grouped["Production"] / grouped["Area_Ha"].replace(0, np.nan)
    return grouped


def run_rainfall_yield_correlations(data, output_root, minimum_pairs=8):
    """Compute Pearson and Spearman associations for each state/crop/unit series."""
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    annual = build_state_year_crop_yield(data)
    results = []

    for keys, group in annual.groupby(["State", "Crop", "Production_Unit"]):
        state, crop, unit = keys
        for rainfall_column, label in [
            ("Annual_Rainfall_mm", "Annual rainfall"),
            ("JJAS_Rainfall_mm", "Monsoon (JJAS) rainfall"),
        ]:
            pair = group[[rainfall_column, "Weighted_Yield"]].dropna()
            if len(pair) < minimum_pairs:
                continue
            rainfall = pair[rainfall_column].to_numpy(dtype=float)
            yield_values = pair["Weighted_Yield"].to_numpy(dtype=float)
            if np.std(rainfall) == 0 or np.std(yield_values) == 0:
                continue
            pearson = stats.pearsonr(rainfall, yield_values)
            spearman = stats.spearmanr(rainfall, yield_values)
            results.append({
                "State": state,
                "Crop": crop,
                "Production_Unit": unit,
                "Rainfall_Measure": label,
                "Sample_Years": len(pair),
                "Pearson_r": float(pearson.statistic),
                "Pearson_p_value": float(pearson.pvalue),
                "Spearman_rho": float(spearman.statistic),
                "Spearman_p_value": float(spearman.pvalue),
                "Interpretation_Guide": "Association only; not proof of causation.",
            })
    result_table = pd.DataFrame(results)
    if not result_table.empty:
        result_table["Absolute_Pearson_r"] = result_table["Pearson_r"].abs()
        result_table = result_table.sort_values("Absolute_Pearson_r", ascending=False)
    result_table.to_csv(os.path.join(tables_dir, "rainfall_yield_correlations.csv"), index=False)
    return result_table


def run_season_hypothesis_test(data, state, crop, output_root, alpha=0.05):
    """Compare yield distributions across seasons for one selected state and crop.

    Kruskal-Wallis is used as a non-parametric comparison. Interpret this as an
    exploratory test: rows share years/locations and are not a causal experiment.
    At least five non-missing yield observations per included season are required.
    """
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    subset = data[(data["State"] == state) & (data["Crop"] == crop)].copy()
    subset["Yield"] = pd.to_numeric(subset["Yield"], errors="coerce")
    season_values = {}
    for season, group in subset.groupby("Season"):
        values = group["Yield"].dropna().to_numpy(dtype=float)
        if len(values) >= 5:
            season_values[str(season)] = values

    if len(season_values) < 2:
        raise ValueError(
            "Not enough data for a season comparison. Choose a crop with at least "
            "two seasons, each with at least five non-missing yield observations."
        )

    test_result = stats.kruskal(*season_values.values())
    p_value = float(test_result.pvalue)
    summary = {
        "State": state,
        "Crop": crop,
        "Test": "Kruskal-Wallis H-test",
        "Seasons_Compared": ", ".join(season_values.keys()),
        "Observations_Per_Season": "; ".join(
            season + ": " + str(len(values)) for season, values in season_values.items()
        ),
        "H_Statistic": float(test_result.statistic),
        "P_Value": p_value,
        "Alpha": alpha,
        "Reject_Null": bool(p_value < alpha),
        "Null_Hypothesis": "Yield distributions are the same across the selected seasons.",
        "Alternative_Hypothesis": "At least one selected season has a different yield distribution.",
        "Conclusion": (
            "Evidence of a difference at the selected significance level."
            if p_value < alpha else
            "Insufficient evidence of a difference at the selected significance level."
        ),
        "Caveat": "Exploratory association; repeated locations/years may not be independent; no causal claim.",
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(tables_dir, "latest_season_hypothesis_test.csv"), index=False
    )
    return summary
