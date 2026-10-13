"""Load and audit the KISAN-X dataset without altering the source CSV."""

import os
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Year", "State", "District", "Crop", "Season", "Area_Ha",
    "Production", "Production_Unit", "Yield", "Annual_Rainfall_mm",
    "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal",
]
KEY_COLUMNS = ["Year", "State", "District", "Crop", "Season"]
NUMERIC_COLUMNS = [
    "Area_Ha", "Production", "Yield", "Annual_Rainfall_mm",
    "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal",
]
TEXT_COLUMNS = ["Year", "State", "District", "Crop", "Season", "Production_Unit"]


def load_dataset(file_path):
    """Load the input CSV and validate its required schema.

    The source file is read only. The returned DataFrame is a separate in-memory
    object; this function never writes back to or edits the source CSV.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Dataset not found at: " + str(file_path) +
            "\nPlace the final CSV at data/this_file.csv."
        )

    data = pd.read_csv(file_path, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns) +
            "\nExpected schema: " + ", ".join(REQUIRED_COLUMNS)
        )

    # Convert analytical fields to numeric in memory; invalid text becomes NaN and
    # is reported by the audit rather than silently removed.
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in TEXT_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    return data


def add_year_start(data):
    """Return a copy with Year_Start extracted for ordering and time-series work."""
    result = data.copy()
    result["Year_Start"] = pd.to_numeric(
        result["Year"].astype("string").str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    return result


def make_output_directories(project_root):
    """Create the output folders used by the CLI."""
    output_root = os.path.join(project_root, "outputs")
    for folder in ["tables", "figures", "forecasts"]:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    return output_root


def run_data_audit(data, output_root):
    """Create audit tables for schema, missingness, duplicates and plausibility.

    Missing MSP values are expected because MSP is only available for selected
    crop-years. They are counted and retained, not imputed.
    """
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    key_columns_present = [column for column in KEY_COLUMNS if column in data.columns]
    exact_duplicate_count = int(data.duplicated().sum())
    duplicate_key_count = int(data.duplicated(key_columns_present).sum())
    year_info = add_year_start(data)

    area_negative = int((data["Area_Ha"] < 0).sum())
    production_negative = int((data["Production"] < 0).sum())
    yield_negative = int((data["Yield"] < 0).sum())
    yield_check = data["Production"] / data["Area_Ha"].replace(0, np.nan)
    yield_mismatch = int((~np.isclose(
        yield_check.to_numpy(dtype=float),
        data["Yield"].to_numpy(dtype=float),
        rtol=1e-7, atol=1e-7, equal_nan=True
    )).sum())

    # Rainfall is repeated across district/crop records; it should be unique at
    # State + Year level in this final dataset.
    rain_key = year_info.groupby(["State", "Year_Start"])[
        ["Annual_Rainfall_mm", "JJAS_Rainfall_mm"]
    ].nunique(dropna=True)
    rainfall_key_conflicts = int((rain_key > 1).any(axis=1).sum())

    audit_rows = [
        ("Rows", len(data), "Number of observations loaded"),
        ("Columns", len(data.columns), "Number of columns in the loaded CSV"),
        ("Required columns present", all(c in data.columns for c in REQUIRED_COLUMNS), "All 12 project columns are required"),
        ("Exact duplicate rows", exact_duplicate_count, "Expected 0; counted, not silently deleted"),
        ("Duplicate analytical keys", duplicate_key_count, "Key: Year + State + District + Crop + Season; expected 0"),
        ("States", int(data["State"].nunique(dropna=True)), "Unique states/UTs represented"),
        ("Districts", int(data["District"].nunique(dropna=True)), "Unique district labels represented"),
        ("Crops", int(data["Crop"].nunique(dropna=True)), "Unique crop labels represented"),
        ("Seasons", int(data["Season"].nunique(dropna=True)), "Unique crop-season labels represented"),
        ("Production units", int(data["Production_Unit"].nunique(dropna=True)), "Production must only be aggregated within compatible units"),
        ("First agricultural year", str(data.loc[year_info["Year_Start"].idxmin(), "Year"]) if year_info["Year_Start"].notna().any() else "Unavailable", "Derived from year labels"),
        ("Last agricultural year", str(data.loc[year_info["Year_Start"].idxmax(), "Year"]) if year_info["Year_Start"].notna().any() else "Unavailable", "Derived from year labels"),
        ("Missing MSP values", int(data["MSP_Rs_per_Quintal"].isna().sum()), "Expected source-coverage gaps; not replaced with zero"),
        ("Negative area values", area_negative, "Expected 0"),
        ("Negative production values", production_negative, "Expected 0"),
        ("Negative yield values", yield_negative, "Expected 0"),
        ("Yield inconsistent with Production / Area", yield_mismatch, "Floating-point tolerance 1e-7; verify if non-zero"),
        ("State-year rainfall conflicts", rainfall_key_conflicts, "Expected 0: rainfall proxy should have one value per State + Year"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["Metric", "Value", "Interpretation"])
    audit.to_csv(os.path.join(tables_dir, "data_audit_summary.csv"), index=False)

    missing_rows = []
    for column in data.columns:
        missing_count = int(data[column].isna().sum())
        missing_rows.append({
            "Column": column,
            "Missing_Count": missing_count,
            "Missing_Percent": round(missing_count / len(data) * 100, 4) if len(data) else 0,
            "Note": "Expected coverage gap" if column == "MSP_Rs_per_Quintal" and missing_count else "Review if non-zero",
        })
    missing_table = pd.DataFrame(missing_rows).sort_values(
        ["Missing_Count", "Column"], ascending=[False, True]
    )
    missing_table.to_csv(os.path.join(tables_dir, "missing_values_by_column.csv"), index=False)

    unit_table = data.groupby("Production_Unit", dropna=False).agg(
        Records=("Production", "size"),
        Distinct_Crops=("Crop", "nunique"),
        Total_Reported_Area_Ha=("Area_Ha", "sum"),
    ).reset_index()
    unit_table.to_csv(os.path.join(tables_dir, "production_unit_coverage.csv"), index=False)

    return audit, missing_table, unit_table
