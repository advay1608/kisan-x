"""Exploratory and descriptive summaries for KISAN-X."""

import os
import numpy as np
import pandas as pd
from data_utils import add_year_start


def build_eda_tables(data, output_root):
    """Build core state, crop, time, unit and yield summary tables."""
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    work = add_year_start(data)

    # This is the sum of reported crop-season area across district records, not
    # a claim about unique physical land area (a field may support multiple crops
    # or seasons during a year).
    area_by_state_year = work.groupby(
        ["Year_Start", "Year", "State"], as_index=False
    ).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"),
        Records=("Area_Ha", "size"),
    ).sort_values(["State", "Year_Start"])
    area_by_state_year.to_csv(os.path.join(tables_dir, "area_by_state_year.csv"), index=False)

    # Production is aggregated only within compatible units.
    production_by_state_year_unit = work.groupby(
        ["Year_Start", "Year", "State", "Production_Unit"], as_index=False
    ).agg(
        Production_Total=("Production", "sum"),
        Records=("Production", "size"),
    ).sort_values(["State", "Production_Unit", "Year_Start"])
    production_by_state_year_unit.to_csv(
        os.path.join(tables_dir, "production_by_state_year_unit.csv"), index=False
    )

    crop_area_by_state = work.groupby(
        ["State", "Crop"], as_index=False
    ).agg(
        Reported_Crop_Area_Ha=("Area_Ha", "sum"),
        Records=("Area_Ha", "size"),
    ).sort_values(["State", "Reported_Crop_Area_Ha"], ascending=[True, False])
    crop_area_by_state.to_csv(os.path.join(tables_dir, "crop_area_by_state.csv"), index=False)

    yield_summary = work.groupby(
        ["State", "Crop", "Production_Unit"], as_index=False
    ).agg(
        Records=("Yield", "count"),
        Mean_Yield=("Yield", "mean"),
        Median_Yield=("Yield", "median"),
        Std_Yield=("Yield", "std"),
        Min_Yield=("Yield", "min"),
        Max_Yield=("Yield", "max"),
    ).sort_values(["State", "Crop", "Production_Unit"])
    yield_summary.to_csv(os.path.join(tables_dir, "yield_summary_by_state_crop.csv"), index=False)

    top_crop_rows = []
    for state, group in crop_area_by_state.groupby("State"):
        top = group.nlargest(10, "Reported_Crop_Area_Ha").copy()
        top["Rank_By_Reported_Area"] = range(1, len(top) + 1)
        top_crop_rows.append(top)
    top_crops = pd.concat(top_crop_rows, ignore_index=True) if top_crop_rows else pd.DataFrame()
    top_crops.to_csv(os.path.join(tables_dir, "top_crops_by_state_area.csv"), index=False)

    # Crop totals are annual and unit-specific; this table supports state/year
    # and crop-level trend plots without mixing Bales with Tonnes.
    production_by_crop_year = work.groupby(
        ["Year_Start", "Year", "State", "Crop", "Production_Unit"], as_index=False
    ).agg(
        Production_Total=("Production", "sum"),
        Reported_Crop_Area_Ha=("Area_Ha", "sum"),
        Weighted_Yield=("Production", "sum"),
    )
    production_by_crop_year["Weighted_Yield"] = (
        production_by_crop_year["Weighted_Yield"] /
        production_by_crop_year["Reported_Crop_Area_Ha"].replace(0, np.nan)
    )
    production_by_crop_year.to_csv(
        os.path.join(tables_dir, "production_by_crop_year.csv"), index=False
    )

    return {
        "area_by_state_year": area_by_state_year,
        "production_by_state_year_unit": production_by_state_year_unit,
        "crop_area_by_state": crop_area_by_state,
        "yield_summary": yield_summary,
        "top_crops": top_crops,
        "production_by_crop_year": production_by_crop_year,
    }
