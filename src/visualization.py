"""Matplotlib and Seaborn figures saved for reports and viva demonstrations."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data_utils import add_year_start

sns.set_theme(style="whitegrid", context="notebook")


def make_eda_plots(data, tables, output_root):
    """Save overview plots for area, production, crop area and yield distribution."""
    figures_dir = os.path.join(output_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    paths = []

    area = tables["area_by_state_year"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(data=area, x="Year_Start", y="Reported_Crop_Area_Ha", hue="State", marker="o", ax=ax)
    ax.set_title("Reported Crop Area by State and Agricultural Year")
    ax.set_xlabel("Agricultural year (starting year)")
    ax.set_ylabel("Sum of reported crop-season area (ha)")
    fig.tight_layout()
    path = os.path.join(figures_dir, "reported_crop_area_trends_by_state.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    production = tables["production_by_state_year_unit"]
    for unit, group in production.groupby("Production_Unit"):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.lineplot(data=group, x="Year_Start", y="Production_Total", hue="State", marker="o", ax=ax)
        ax.set_title("Reported Production Trend by State (unit: " + str(unit) + ")")
        ax.set_xlabel("Agricultural year (starting year)")
        ax.set_ylabel("Production (" + str(unit) + ")")
        fig.tight_layout()
        name = "production_trends_" + str(unit).replace("/", "-").replace(" ", "_") + ".png"
        path = os.path.join(figures_dir, name)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    for state, group in tables["crop_area_by_state"].groupby("State"):
        top = group.nlargest(10, "Reported_Crop_Area_Ha").sort_values("Reported_Crop_Area_Ha")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=top, x="Reported_Crop_Area_Ha", y="Crop", color="steelblue", ax=ax)
        ax.set_title("Top Crops by Reported Area — " + str(state))
        ax.set_xlabel("Sum of reported crop-season area (ha)")
        ax.set_ylabel("Crop")
        fig.tight_layout()
        name = "top_crops_area_" + str(state).replace(" ", "_") + ".png"
        path = os.path.join(figures_dir, name)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    tonne_yield = data.loc[
        (data["Production_Unit"] == "Tonnes") & data["Yield"].notna(), "Yield"
    ]
    if len(tonne_yield):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.histplot(tonne_yield, bins=40, kde=True, color="steelblue", ax=ax)
        ax.set_title("Yield Distribution for Records Measured in Tonnes")
        ax.set_xlabel("Yield (tonnes per hectare, as reported/derived in the source)")
        ax.set_ylabel("Record count")
        fig.tight_layout()
        path = os.path.join(figures_dir, "yield_distribution_tonnes.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    # Add one representative rainfall/yield scatter per state, selecting the
    # crop with the longest annual history. Each plot is still crop-specific.
    try:
        from statistical_analysis import build_state_year_crop_yield
        annual_yield = build_state_year_crop_yield(data)
        for state, state_group in annual_yield.groupby("State"):
            series_counts = state_group.groupby(["Crop", "Production_Unit"])["Year_Start"].nunique()
            if series_counts.empty:
                continue
            crop, unit = series_counts.sort_values(ascending=False).index[0]
            if int(series_counts.loc[(crop, unit)]) >= 8:
                path = plot_rainfall_yield_scatter(data, state, crop, output_root)
                paths.append(path)
    except (ValueError, KeyError, IndexError):
        # The main analysis outputs are still useful if no suitable plot exists.
        pass

    return paths


def plot_rainfall_yield_scatter(data, state, crop, output_root):
    """Plot annual monsoon rainfall against state-level weighted yield for a crop."""
    from statistical_analysis import build_state_year_crop_yield
    figures_dir = os.path.join(output_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    annual = build_state_year_crop_yield(data)
    subset = annual[(annual["State"] == state) & (annual["Crop"] == crop)].copy()
    if subset.empty:
        raise ValueError("No annual series found for the selected state and crop.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=subset, x="JJAS_Rainfall_mm", y="Weighted_Yield", hue="Production_Unit", s=75, ax=ax)
    ax.set_title("Monsoon Rainfall and Crop Yield — " + state + " / " + crop)
    ax.set_xlabel("JJAS rainfall proxy (mm)")
    ax.set_ylabel("Weighted yield (production unit per hectare)")
    fig.tight_layout()
    safe_name = (state + "_" + crop).replace("/", "-").replace(" ", "_")
    path = os.path.join(figures_dir, "rainfall_yield_" + safe_name + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_forecast_comparison(annual, train, test, model_predictions, future_starts, future_values,
                             output_path, state, crop, production_unit):
    """Create a simple historical/holdout/future forecast chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_years = annual["Year_Start"].astype(int).to_numpy()
    test_years = all_years[-len(test):]
    history_years = all_years[:-len(test)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(history_years, train, marker="o", label="Training observations")
    ax.plot(test_years, test, marker="o", linestyle="--", label="Actual holdout")
    ax.plot(test_years, model_predictions, marker="x", linestyle="--", label="ARIMA holdout forecast")
    ax.plot(future_starts, future_values, marker="o", linestyle=":", label="Future ARIMA forecast")
    ax.set_title("Crop Production Forecast — " + state + " / " + crop)
    ax.set_xlabel("Agricultural year (starting year)")
    ax.set_ylabel("Production (" + production_unit + ")")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_diversification_metrics(metrics, output_root):
    """Plot the area diversity indicator and HHI concentration by state/year."""
    figures_dir = os.path.join(output_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    paths = []
    for column, title, ylabel, filename in [
        ("Area_Diversity_Index", "Crop-Area Diversity by State and Year", "1 - HHI (higher means more even area shares)", "area_diversity_by_state.png"),
        ("HHI_Area_Concentration", "Crop-Area Concentration by State and Year", "HHI (higher means more concentrated area shares)", "area_concentration_by_state.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.lineplot(data=metrics, x="Year_Start", y=column, hue="State", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Agricultural year (starting year)")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        path = os.path.join(figures_dir, filename)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
