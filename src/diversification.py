"""Crop-area shares and simple crop concentration/diversity measures."""

import os
import pandas as pd
from data_utils import add_year_start


def run_diversification_analysis(data, output_root):
    """Calculate reported crop-area shares and area concentration by state/year.

    HHI = sum of squared crop area shares (shares are between 0 and 1).
    A larger HHI means more concentration in fewer crops. 1 - HHI is a simple
    diversity indicator, and 1 / HHI is the effective number of equally sized
    crops. Area is the reported sum across crop/season records, not unique land.
    """
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    work = add_year_start(data)

    crop_area = work.groupby(
        ["Year_Start", "Year", "State", "Crop"], as_index=False
    ).agg(Reported_Crop_Area_Ha=("Area_Ha", "sum"))
    totals = crop_area.groupby(["Year_Start", "Year", "State"])[
        "Reported_Crop_Area_Ha"
    ].transform("sum")
    crop_area["Area_Share"] = crop_area["Reported_Crop_Area_Ha"] / totals
    crop_area["Area_Share_Percent"] = crop_area["Area_Share"] * 100
    crop_area.to_csv(os.path.join(tables_dir, "crop_area_shares.csv"), index=False)

    crop_area["Squared_Area_Share"] = crop_area["Area_Share"] ** 2
    metrics = crop_area.groupby(["Year_Start", "Year", "State"], as_index=False).agg(
        Number_of_Crops=("Crop", "nunique"),
        HHI_Area_Concentration=("Squared_Area_Share", "sum"),
        Total_Reported_Crop_Area_Ha=("Reported_Crop_Area_Ha", "sum"),
    )
    metrics["Area_Diversity_Index"] = 1 - metrics["HHI_Area_Concentration"]
    metrics["Effective_Number_of_Crops"] = 1 / metrics["HHI_Area_Concentration"].replace(0, float("nan"))
    metrics.to_csv(os.path.join(tables_dir, "diversification_by_state_year.csv"), index=False)

    top_crops = crop_area.sort_values(
        ["State", "Year_Start", "Area_Share_Percent"], ascending=[True, True, False]
    ).groupby(["Year_Start", "State"], as_index=False).head(5)
    top_crops.to_csv(os.path.join(tables_dir, "top_five_crop_shares_by_year.csv"), index=False)
    return crop_area, metrics, top_crops
