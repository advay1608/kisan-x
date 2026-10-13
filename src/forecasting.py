"""Introductory time-series forecasting using Statsmodels ARIMA."""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from data_utils import add_year_start


def get_complete_series_candidates(data, state=None, minimum_years=15):
    """List state/crop/unit series with consecutive annual production observations."""
    work = add_year_start(data)
    if state is not None:
        work = work[work["State"] == state]
    annual = work.groupby(
        ["Year_Start", "Year", "State", "Crop", "Production_Unit"], as_index=False
    ).agg(
        Production=("Production", "sum"),
        Reported_Crop_Area_Ha=("Area_Ha", "sum"),
    )
    candidates = []
    for keys, group in annual.groupby(["State", "Crop", "Production_Unit"]):
        group = group.sort_values("Year_Start")
        years = group["Year_Start"].astype(int).to_numpy()
        if len(years) < minimum_years or len(np.unique(years)) != len(years):
            continue
        if len(years) > 1 and not np.all(np.diff(years) == 1):
            continue
        candidates.append({
            "State": keys[0],
            "Crop": keys[1],
            "Production_Unit": keys[2],
            "Annual_Points": len(group),
            "First_Year": str(group.iloc[0]["Year"]),
            "Last_Year": str(group.iloc[-1]["Year"]),
            "Average_Annual_Reported_Area_Ha": float(group["Reported_Crop_Area_Ha"].mean()),
        })
    result = pd.DataFrame(candidates)
    if not result.empty:
        result = result.sort_values(["State", "Average_Annual_Reported_Area_Ha"], ascending=[True, False])
    return result


def _agricultural_year_label(year_start):
    """Format integer start year as 1997-98 style agricultural-year label."""
    return str(int(year_start)) + "-" + str((int(year_start) + 1) % 100).zfill(2)


def get_annual_production_series(data, state, crop, production_unit):
    """Aggregate production across districts/seasons for one state/crop/unit."""
    work = add_year_start(data)
    subset = work[
        (work["State"] == state) &
        (work["Crop"] == crop) &
        (work["Production_Unit"] == production_unit)
    ].copy()
    if subset.empty:
        raise ValueError("No records found for the selected state, crop and production unit.")
    annual = subset.groupby(["Year_Start", "Year"], as_index=False).agg(
        Production=("Production", "sum"),
        Reported_Crop_Area_Ha=("Area_Ha", "sum"),
    ).sort_values("Year_Start")
    years = annual["Year_Start"].astype(int).to_numpy()
    if len(years) < 15:
        raise ValueError("At least 15 annual points are required for this educational forecast.")
    if len(np.unique(years)) != len(years) or not np.all(np.diff(years) == 1):
        raise ValueError(
            "This series has missing annual observations. KISAN-X will not fill missing "
            "years with zero; choose a complete state/crop/unit series instead."
        )
    if annual["Production"].nunique() < 3:
        raise ValueError("The selected production series has too little variation to forecast usefully.")
    return annual


def _error_metrics(actual, predicted):
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def forecast_crop_production(data, state, crop, production_unit, output_root, holdout_years=3, future_years=3):
    """Fit ARIMA(1,1,0), evaluate chronologically, compare with a naive baseline,
    refit on the full series and forecast the next three agricultural years.

    ARIMA(1,1,0) is intentionally fixed and explainable rather than selected by
    a complex automatic model-search framework. Model output is an estimate,
    not a guarantee. Missing years are never fabricated.
    """
    forecasts_dir = os.path.join(output_root, "forecasts")
    figures_dir = os.path.join(output_root, "figures")
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(forecasts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    annual = get_annual_production_series(data, state, crop, production_unit)
    values = annual["Production"].to_numpy(dtype=float)
    year_starts = annual["Year_Start"].astype(int).to_numpy()
    holdout_years = min(max(2, int(holdout_years)), 4, len(values) - 12)
    train = values[:-holdout_years]
    test = values[-holdout_years:]
    test_year_labels = annual["Year"].iloc[-holdout_years:].astype(str).tolist()

    adf_statistic = np.nan
    adf_p_value = np.nan
    adf_note = "Not computed"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(train, autolag="AIC")
        adf_statistic = float(adf_result[0])
        adf_p_value = float(adf_result[1])
        adf_note = (
            "Evidence against a unit root at 5% in the training series."
            if adf_p_value < 0.05 else
            "Insufficient evidence against a unit root at 5%; differencing may be useful."
        )
    except Exception as error:
        adf_note = "Stationarity test unavailable: " + str(error)

    naive_predictions = np.concatenate(([train[-1]], test[:-1]))
    naive_mae, naive_rmse = _error_metrics(test, naive_predictions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluation_model = ARIMA(
            train, order=(1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        model_predictions = np.asarray(evaluation_model.forecast(steps=holdout_years), dtype=float)
    arima_mae, arima_rmse = _error_metrics(test, model_predictions)

    # Fit on all known years after evaluating the holdout period.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = ARIMA(
            values, order=(1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        future_object = final_model.get_forecast(steps=future_years)
        raw_future = np.asarray(future_object.predicted_mean, dtype=float)
        confidence = np.asarray(future_object.conf_int(alpha=0.05), dtype=float)

    # Production cannot be negative. Preserve the raw statistical prediction as
    # well as the non-negative presentation value so the constraint is visible.
    future_starts = np.arange(year_starts[-1] + 1, year_starts[-1] + 1 + future_years)
    future_rows = []
    for index, year_start in enumerate(future_starts):
        future_rows.append({
            "State": state,
            "Crop": crop,
            "Production_Unit": production_unit,
            "Forecast_Year": _agricultural_year_label(year_start),
            "ARIMA_Raw_Forecast": float(raw_future[index]),
            "Forecast_Production_NonNegative": float(max(0, raw_future[index])),
            "Lower_95_Percent_Confidence": float(confidence[index, 0]),
            "Upper_95_Percent_Confidence": float(confidence[index, 1]),
            "Model": "ARIMA(1,1,0)",
            "Note": "Model estimate, not a guarantee; future rainfall/MSP are not inputs.",
        })
    future_table = pd.DataFrame(future_rows)

    evaluation_rows = []
    for label, mae, rmse in [
        ("Naive previous-year baseline", naive_mae, naive_rmse),
        ("ARIMA(1,1,0)", arima_mae, arima_rmse),
    ]:
        evaluation_rows.append({
            "State": state,
            "Crop": crop,
            "Production_Unit": production_unit,
            "Model": label,
            "Training_Years": len(train),
            "Holdout_Years": len(test),
            "Holdout_First_Year": test_year_labels[0],
            "Holdout_Last_Year": test_year_labels[-1],
            "MAE": mae,
            "RMSE": rmse,
            "ADF_Statistic_Training": adf_statistic,
            "ADF_P_Value_Training": adf_p_value,
            "Stationarity_Note": adf_note,
            "ARIMA_Order": "(1,1,0)" if label == "ARIMA(1,1,0)" else "Not applicable",
        })
    evaluation_table = pd.DataFrame(evaluation_rows)

    safe_name = "_".join([state, crop, production_unit]).replace("/", "-").replace(" ", "_")
    future_table.to_csv(os.path.join(forecasts_dir, safe_name + "_future_forecast.csv"), index=False)
    evaluation_table.to_csv(os.path.join(forecasts_dir, safe_name + "_evaluation.csv"), index=False)

    # Re-create the comparison chart through the shared visualization module.
    from visualization import plot_forecast_comparison
    plot_forecast_comparison(
        annual, train, test, model_predictions, future_starts, raw_future,
        os.path.join(figures_dir, safe_name + "_forecast.png"),
        state, crop, production_unit,
    )

    return {
        "annual_series": annual,
        "evaluation": evaluation_table,
        "future_forecast": future_table,
        "adf_statistic": adf_statistic,
        "adf_p_value": adf_p_value,
        "arima_mae": arima_mae,
        "arima_rmse": arima_rmse,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "selected_state": state,
        "selected_crop": crop,
        "selected_unit": production_unit,
    }
