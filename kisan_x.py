"""KISAN-X analysis module. main.py shows the menu and calls these functions."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid")


# ---------------- Settings ----------------

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "this_file.csv")
OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, "outputs")
TABLE_FOLDER = os.path.join(OUTPUT_FOLDER, "tables")
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figures")
FORECAST_FOLDER = os.path.join(OUTPUT_FOLDER, "forecasts")
REPORT_FILE = os.path.join(OUTPUT_FOLDER, "KISAN_X_ANALYSIS_REPORT.md")

COLUMNS = ["Year", "State", "District", "Crop", "Season", "Area_Ha", "Production",
           "Production_Unit", "Yield", "Annual_Rainfall_mm", "JJAS_Rainfall_mm",
           "MSP_Rs_per_Quintal"]
NUMBER_COLUMNS = ["Area_Ha", "Production", "Yield", "Annual_Rainfall_mm",
                  "JJAS_Rainfall_mm", "MSP_Rs_per_Quintal"]
TEXT_COLUMNS = ["Year", "State", "District", "Crop", "Season", "Production_Unit"]
RAINFALL_COLUMNS = ["Annual_Rainfall_mm", "JJAS_Rainfall_mm"]

MIN_YEARS_FOR_CORRELATION = 8
MIN_VALUES_PER_SEASON = 5
MIN_YEARS_FOR_FORECAST = 15
TEST_YEARS = 3
FUTURE_YEARS = 3


# ---------------- Saving files ----------------

def make_output_folders():
    for folder in [TABLE_FOLDER, FIGURE_FOLDER, FORECAST_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def save_table(table, file_name, folder=TABLE_FOLDER):
    path = os.path.join(folder, file_name)
    table.to_csv(path, index=False)
    return path


def save_chart(figure, file_name):
    path = os.path.join(FIGURE_FOLDER, file_name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def clean_name(text):
    return str(text).replace("/", "-").replace(" ", "_")


def line_chart_by_state(table, y_column, title, y_label, file_name):
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=table, x="Year_Start", y=y_column, hue="State", marker="o", ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel(y_label)
    return save_chart(figure, file_name)


# ---------------- Loading the data ----------------

def load_data():
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError("Dataset not found: " + DATA_FILE)

    data = pd.read_csv(DATA_FILE)

    missing_columns = []
    for column in COLUMNS:
        if column not in data.columns:
            missing_columns.append(column)
    if missing_columns:
        raise ValueError("These columns are missing from the dataset: " + ", ".join(missing_columns))

    for column in NUMBER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in TEXT_COLUMNS:
        data[column] = data[column].astype(str).str.strip()

    data["Year_Start"] = data["Year"].str[:4].astype(int)
    return data


def yearly_totals(data):
    yearly = data.groupby(["State", "Crop", "Production_Unit", "Year_Start", "Year"],
                          as_index=False).agg(
        Area_Ha=("Area_Ha", "sum"),
        Production=("Production", "sum"),
        Annual_Rainfall_mm=("Annual_Rainfall_mm", "first"),
        JJAS_Rainfall_mm=("JJAS_Rainfall_mm", "first"),
    )
    yearly["Weighted_Yield"] = yearly["Production"] / yearly["Area_Ha"].replace(0, np.nan)
    return yearly


# ---------------- 1. Data quality ----------------

def find_yield_outliers(data):
    yield_by_crop = data.groupby(["Crop", "Production_Unit"])["Yield"]
    q1 = yield_by_crop.transform("quantile", 0.25)
    q3 = yield_by_crop.transform("quantile", 0.75)
    iqr = q3 - q1
    is_outlier = (data["Yield"] < q1 - 1.5 * iqr) | (data["Yield"] > q3 + 1.5 * iqr)
    return data.loc[is_outlier, ["Year", "State", "District", "Crop", "Season",
                                 "Production_Unit", "Yield"]]


def check_data_quality(data):
    calculated_yield = data["Production"] / data["Area_Ha"].replace(0, np.nan)
    yield_matches = np.isclose(calculated_yield, data["Yield"], equal_nan=True)
    rainfall_values_per_state_year = data.groupby(["State", "Year"])["Annual_Rainfall_mm"].nunique()
    outliers = find_yield_outliers(data)

    checks = [
        ["Rows", len(data), "All rows kept"],
        ["States", data["State"].nunique(), ""],
        ["Districts", data["District"].nunique(), ""],
        ["Crops", data["Crop"].nunique(), ""],
        ["Years", data["Year"].nunique(), data["Year"].min() + " to " + data["Year"].max()],
        ["Exact duplicate rows", data.duplicated().sum(), "Counted, not deleted"],
        ["Duplicate Year+State+District+Crop+Season",
         data.duplicated(["Year", "State", "District", "Crop", "Season"]).sum(), "Counted, not deleted"],
        ["Missing MSP values", data["MSP_Rs_per_Quintal"].isna().sum(),
         "Left empty: no MSP is announced for these crops/years, it is not a zero price"],
        ["Negative area / production / yield",
         (data[["Area_Ha", "Production", "Yield"]] < 0).sum().sum(), "Should be 0"],
        ["Yield not equal to Production / Area", (~yield_matches).sum(), "Should be 0"],
        ["Most rainfall values in one state-year", rainfall_values_per_state_year.max(),
         "1 means rainfall is one state value copied to every district"],
        ["Yield outliers (1.5 x IQR within each crop)", len(outliers),
         "Kept: extreme yields can be real (drought or bumper year)"],
    ]
    quality = pd.DataFrame(checks, columns=["Check", "Value", "Action / Meaning"])

    missing = pd.DataFrame({
        "Column": COLUMNS,
        "Missing": data[COLUMNS].isna().sum().values,
    })
    missing["Missing_Percent"] = (missing["Missing"] / len(data) * 100).round(2)

    units = data.groupby("Production_Unit", as_index=False).agg(
        Records=("Crop", "count"), Crops=("Crop", "nunique"))

    save_table(quality, "data_quality.csv")
    save_table(missing, "missing_values.csv")
    save_table(outliers, "yield_outliers.csv")
    save_table(units, "production_units.csv")
    return quality, missing, units


# ---------------- 2. Exploratory analysis ----------------

def explore_data(data):
    area = data.groupby(["State", "Year_Start"], as_index=False)["Area_Ha"].sum()
    production = data.groupby(["State", "Production_Unit", "Year_Start"],
                              as_index=False)["Production"].sum()
    crop_area = data.groupby(["State", "Crop"], as_index=False)["Area_Ha"].sum()
    crop_area = crop_area.sort_values(["State", "Area_Ha"], ascending=[True, False])
    top_crops = crop_area.groupby("State").head(10)

    save_table(area, "area_by_state_year.csv")
    save_table(production, "production_by_state_year_unit.csv")
    save_table(top_crops, "top_10_crops_by_area.csv")

    charts = [line_chart_by_state(area, "Area_Ha", "Reported Crop Area by State",
                                  "Reported crop-season area (ha)", "area_trend.png")]

    for unit, unit_rows in production.groupby("Production_Unit"):
        charts.append(line_chart_by_state(unit_rows, "Production", f"Production by State ({unit})",
                                          f"Production ({unit})", f"production_trend_{unit}.png"))

    for state, state_rows in top_crops.groupby("State"):
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.barplot(data=state_rows, x="Area_Ha", y="Crop", color="seagreen", ax=axis)
        axis.set_title(f"Top 10 Crops by Area - {state}")
        axis.set_xlabel("Reported crop-season area (ha), 1997-2017")
        axis.set_ylabel("")
        charts.append(save_chart(figure, f"top_crops_{clean_name(state)}.png"))

    tonne_yields = data.loc[data["Production_Unit"] == "Tonnes", "Yield"]
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(tonne_yields[tonne_yields < tonne_yields.quantile(0.99)], bins=40, ax=axis)
    axis.set_title("Distribution of Yield (crops measured in Tonnes, top 1% hidden)")
    axis.set_xlabel("Yield (tonnes per hectare)")
    axis.set_ylabel("Number of records")
    charts.append(save_chart(figure, "yield_histogram.png"))

    return top_crops, production, charts


# ---------------- 3. Descriptive statistics ----------------

def describe_data(data):
    yield_stats = data.groupby(["State", "Crop", "Production_Unit"], as_index=False).agg(
        Records=("Yield", "count"),
        Mean=("Yield", "mean"),
        Median=("Yield", "median"),
        Std_Dev=("Yield", "std"),
        Minimum=("Yield", "min"),
        Maximum=("Yield", "max"),
        Skewness=("Yield", "skew"),
    )
    yield_stats["CV_Percent"] = yield_stats["Std_Dev"] / yield_stats["Mean"] * 100
    yield_stats = yield_stats.round(3)

    msp = data[["Crop", "Year_Start", "MSP_Rs_per_Quintal"]].dropna().drop_duplicates()
    msp_stats = msp.groupby("Crop", as_index=False).agg(
        Years=("Year_Start", "count"),
        First_MSP=("MSP_Rs_per_Quintal", "first"),
        Last_MSP=("MSP_Rs_per_Quintal", "last"),
        Mean_MSP=("MSP_Rs_per_Quintal", "mean"),
    )
    msp_stats["Percent_Increase"] = ((msp_stats["Last_MSP"] / msp_stats["First_MSP"] - 1) * 100).round(1)
    msp_stats = msp_stats.sort_values("Years", ascending=False)

    save_table(yield_stats, "yield_statistics.csv")
    save_table(msp_stats, "msp_statistics.csv")

    main_msp_crops = msp_stats["Crop"].head(6)
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=msp[msp["Crop"].isin(main_msp_crops)], x="Year_Start",
                 y="MSP_Rs_per_Quintal", hue="Crop", marker="o", ax=axis)
    axis.set_title("Minimum Support Price Over Time")
    axis.set_xlabel("Year")
    axis.set_ylabel("MSP (Rs per quintal)")
    msp_chart = save_chart(figure, "msp_trend.png")

    return yield_stats, msp_stats, msp_chart


# ---------------- 4. Rainfall and yield ----------------

def correlation_strength(r):
    size = abs(r)
    if size >= 0.7:
        strength = "Strong"
    elif size >= 0.4:
        strength = "Moderate"
    elif size >= 0.2:
        strength = "Weak"
    else:
        return "None"
    if r > 0:
        return strength + " positive"
    return strength + " negative"


def rainfall_yield_correlation(data):
    yearly = yearly_totals(data)
    rows = []
    for (state, crop, unit), group in yearly.groupby(["State", "Crop", "Production_Unit"]):
        for rain_column in RAINFALL_COLUMNS:
            pair = group[[rain_column, "Weighted_Yield"]].dropna()
            if len(pair) < MIN_YEARS_FOR_CORRELATION:
                continue
            if pair[rain_column].nunique() < 2 or pair["Weighted_Yield"].nunique() < 2:
                continue
            pearson_r, pearson_p = stats.pearsonr(pair[rain_column], pair["Weighted_Yield"])
            spearman_rho, spearman_p = stats.spearmanr(pair[rain_column], pair["Weighted_Yield"])
            rows.append({
                "State": state, "Crop": crop, "Production_Unit": unit,
                "Rainfall": rain_column, "Years": len(pair),
                "Pearson_r": round(pearson_r, 3), "Pearson_p": round(pearson_p, 4),
                "Spearman_rho": round(spearman_rho, 3), "Spearman_p": round(spearman_p, 4),
                "Strength": correlation_strength(pearson_r),
                "Significant": pearson_p < 0.05,
            })

    results = pd.DataFrame(rows)
    if not results.empty:
        results["Abs_r"] = results["Pearson_r"].abs()
        results = results.sort_values("Abs_r", ascending=False).drop(columns="Abs_r")
    save_table(results, "rainfall_yield_correlation.csv")
    return results


def plot_rainfall_yield(data, state, crop):
    yearly = yearly_totals(data)
    subset = yearly[(yearly["State"] == state) & (yearly["Crop"] == crop)]
    if subset.empty:
        raise ValueError(f"No records for {crop} in {state}.")
    unit = subset["Production_Unit"].iloc[0]
    subset = subset[subset["Production_Unit"] == unit]

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.regplot(data=subset, x="JJAS_Rainfall_mm", y="Weighted_Yield", ax=axis)
    axis.set_title(f"Monsoon Rainfall vs Yield - {crop}, {state}")
    axis.set_xlabel("June-September rainfall (mm)")
    axis.set_ylabel(f"Yield ({unit} per hectare)")
    return save_chart(figure, f"rainfall_vs_yield_{clean_name(state)}_{clean_name(crop)}.png")


# ---------------- 5. Season comparison (hypothesis test) ----------------

def compare_seasons(data, state, crop):
    subset = data[(data["State"] == state) & (data["Crop"] == crop)]

    yields_by_season = {}
    for season, season_rows in subset.groupby("Season"):
        values = season_rows["Yield"].dropna()
        if len(values) >= MIN_VALUES_PER_SEASON:
            yields_by_season[season] = values

    if len(yields_by_season) < 2:
        raise ValueError(f"{crop} in {state} is not grown in at least two seasons "
                         f"with {MIN_VALUES_PER_SEASON}+ records each.")

    h_statistic, p_value = stats.kruskal(*yields_by_season.values())
    if p_value < 0.05:
        conclusion = "Reject H0: yield differs between seasons"
    else:
        conclusion = "Do not reject H0: no clear yield difference between seasons"

    result = {
        "State": state,
        "Crop": crop,
        "Seasons": ", ".join(yields_by_season.keys()),
        "H_Statistic": round(h_statistic, 3),
        "P_Value": round(p_value, 4),
        "Conclusion": conclusion,
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    compared_rows = subset[subset["Season"].isin(yields_by_season.keys())]
    sns.boxplot(data=compared_rows, x="Season", y="Yield", ax=axis)
    axis.set_title(f"Yield by Season - {crop}, {state} (p = {p_value:.4f})")
    axis.set_ylabel("Yield per hectare")
    result["Chart"] = save_chart(figure, f"season_boxplot_{clean_name(state)}_{clean_name(crop)}.png")

    save_table(pd.DataFrame([result]), "season_test_latest.csv")
    return result


# ---------------- 6. Crop diversification ----------------

def crop_diversity(data):
    crop_area = data.groupby(["State", "Year_Start", "Crop"], as_index=False)["Area_Ha"].sum()
    state_total = crop_area.groupby(["State", "Year_Start"])["Area_Ha"].transform("sum")
    crop_area["Share"] = crop_area["Area_Ha"] / state_total
    crop_area["Share_Squared"] = crop_area["Share"] ** 2

    diversity = crop_area.groupby(["State", "Year_Start"], as_index=False).agg(
        Crops_Grown=("Crop", "count"),
        HHI=("Share_Squared", "sum"),
    )
    diversity["Diversity_Index"] = 1 - diversity["HHI"]
    diversity = diversity.round(4)

    save_table(crop_area.drop(columns="Share_Squared"), "crop_area_shares.csv")
    save_table(diversity, "crop_diversity.csv")
    chart = line_chart_by_state(diversity, "Diversity_Index", "Crop Diversity by State",
                                "Diversity index (1 - HHI)", "crop_diversity.png")
    return diversity, chart


# ---------------- 7. Forecasting ----------------

def forecastable_series(data):
    yearly = yearly_totals(data)
    rows = []
    for (state, crop, unit), group in yearly.groupby(["State", "Crop", "Production_Unit"]):
        years = group["Year_Start"].to_numpy()
        no_missing_years = np.all(np.diff(years) == 1)
        if len(years) >= MIN_YEARS_FOR_FORECAST and no_missing_years:
            rows.append({"State": state, "Crop": crop, "Production_Unit": unit,
                         "Years": len(years), "Average_Area_Ha": round(group["Area_Ha"].mean())})
    series = pd.DataFrame(rows)
    return series.sort_values(["State", "Average_Area_Ha"], ascending=[True, False])


def year_label(year):
    return f"{year}-{(year + 1) % 100:02d}"


def mae_and_rmse(actual, predicted):
    errors = actual - predicted
    return np.mean(np.abs(errors)), np.sqrt(np.mean(errors ** 2))


def fit_arima(values):
    return ARIMA(values, order=(1, 1, 0)).fit()


def forecast_production(data, state, crop, unit):
    yearly = yearly_totals(data)
    series = yearly[(yearly["State"] == state) & (yearly["Crop"] == crop) &
                    (yearly["Production_Unit"] == unit)]
    years = series["Year_Start"].to_numpy()
    production = series["Production"].to_numpy(dtype=float)

    if len(years) < MIN_YEARS_FOR_FORECAST or not np.all(np.diff(years) == 1):
        raise ValueError("This series is too short or has missing years, so it is not forecast.")

    train = production[:-TEST_YEARS]
    test = production[-TEST_YEARS:]

    adf_p_value = adfuller(train, result_object=False)[1]
    if adf_p_value < 0.05:
        stationarity = f"Stationary (ADF p = {adf_p_value:.3f})"
    else:
        stationarity = f"Not stationary (ADF p = {adf_p_value:.3f}), so ARIMA differences it once"

    naive_prediction = production[-TEST_YEARS - 1:-1]
    arima_prediction = fit_arima(train).forecast(TEST_YEARS)
    naive_mae, naive_rmse = mae_and_rmse(test, naive_prediction)
    arima_mae, arima_rmse = mae_and_rmse(test, arima_prediction)

    future = fit_arima(production).get_forecast(FUTURE_YEARS)
    future_values = np.maximum(future.predicted_mean, 0)
    limits = np.maximum(future.conf_int(alpha=0.05), 0)
    future_years = np.arange(years[-1] + 1, years[-1] + 1 + FUTURE_YEARS)

    evaluation = pd.DataFrame({
        "Model": ["Naive (same as last year)", "ARIMA(1,1,0)"],
        "MAE": [round(naive_mae), round(arima_mae)],
        "RMSE": [round(naive_rmse), round(arima_rmse)],
    })
    forecast = pd.DataFrame({
        "Year": [year_label(year) for year in future_years],
        "Forecast": future_values.round(),
        "Lower_95": limits[:, 0].round(),
        "Upper_95": limits[:, 1].round(),
    })

    name = clean_name(f"{state}_{crop}_{unit}")
    save_table(evaluation, f"{name}_evaluation.csv", FORECAST_FOLDER)
    save_table(forecast, f"{name}_forecast.csv", FORECAST_FOLDER)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(years, production, marker="o", label="Actual production")
    axis.plot(years[-TEST_YEARS:], arima_prediction, marker="x", linestyle="--",
              label="ARIMA prediction for test years")
    axis.plot(future_years, future_values, marker="o", linestyle=":", label="Future forecast")
    axis.fill_between(future_years, limits[:, 0], limits[:, 1], alpha=0.2, label="95% range")
    axis.set_title(f"Production Forecast - {crop}, {state}")
    axis.set_xlabel("Year")
    axis.set_ylabel(f"Production ({unit})")
    axis.legend()
    chart = save_chart(figure, f"forecast_{name}.png")

    if arima_mae < naive_mae:
        better_model = "ARIMA(1,1,0)"
    else:
        better_model = "Naive"

    return {"stationarity": stationarity, "evaluation": evaluation, "forecast": forecast,
            "better_model": better_model, "chart": chart}


# ---------------- 8. Full report ----------------

def season_tests_for_report(data, top_crops):
    rows = []
    for state, state_rows in top_crops.groupby("State"):
        for crop in state_rows["Crop"]:
            try:
                result = compare_seasons(data, state, crop)
            except ValueError:
                continue
            rows.append({"State": state, "Crop": crop, "Seasons": result["Seasons"],
                         "P_Value": result["P_Value"], "Conclusion": result["Conclusion"]})
            break
    return pd.DataFrame(rows)


def forecasts_for_report(data):
    rows = []
    for state, state_series in forecastable_series(data).groupby("State"):
        best = state_series.iloc[0]
        result = forecast_production(data, state, best["Crop"], best["Production_Unit"])
        mae = result["evaluation"]["MAE"]
        rows.append({
            "State": state, "Crop": best["Crop"], "Unit": best["Production_Unit"],
            "Naive_MAE": mae[0], "ARIMA_MAE": mae[1], "Better_Model": result["better_model"],
            "Next_Year_Forecast": result["forecast"]["Forecast"].iloc[0],
        })
    return pd.DataFrame(rows)


def production_change(data, top_crops):
    yearly = yearly_totals(data)
    rows = []
    for state, state_rows in top_crops.groupby("State"):
        crop = state_rows["Crop"].iloc[0]
        series = yearly[(yearly["State"] == state) & (yearly["Crop"] == crop)]
        first_average = series["Production"].head(3).mean()
        last_average = series["Production"].tail(3).mean()
        rows.append({
            "State": state, "Crop": crop, "Unit": series["Production_Unit"].iloc[0],
            "First_3_Years_Avg": round(first_average),
            "Last_3_Years_Avg": round(last_average),
            "Percent_Change": round((last_average / first_average - 1) * 100, 1),
        })
    return pd.DataFrame(rows)


def text_table(table):
    return "```\n" + table.to_string(index=False) + "\n```"


def generate_report(data):
    quality, missing, units = check_data_quality(data)
    top_crops, production, charts = explore_data(data)
    yield_stats, msp_stats, msp_chart = describe_data(data)
    correlations = rainfall_yield_correlation(data)
    diversity, diversity_chart = crop_diversity(data)
    season_tests = season_tests_for_report(data, top_crops)
    forecasts = forecasts_for_report(data)
    change = production_change(data, top_crops)

    for state, state_rows in correlations.groupby("State"):
        plot_rainfall_yield(data, state, state_rows["Crop"].iloc[0])

    leading_crops = top_crops.groupby("State").head(3)
    significant = correlations[correlations["Significant"]]
    strongest = correlations.iloc[0]
    latest = diversity[diversity["Year_Start"] == diversity["Year_Start"].max()]
    latest = latest.sort_values("Diversity_Index", ascending=False)
    arima_wins = (forecasts["Better_Model"] == "ARIMA(1,1,0)").sum()

    lines = [
        "# KISAN-X Analysis Report",
        "",
        "## 1. Problem statement",
        "",
        "Using district-level crop records (1997-2017) for Maharashtra, Uttar Pradesh, Rajasthan",
        "and Madhya Pradesh, KISAN-X answers six questions:",
        "",
        "1. Which crops take up the most area in each state?",
        "2. How has crop production changed over time?",
        "3. Is rainfall associated with crop yield?",
        "4. Does yield differ between seasons for the same crop?",
        "5. How diversified is each state's cropping pattern?",
        "6. Can past production be used to forecast future production?",
        "",
        "## 2. Dataset and data quality",
        "",
        text_table(quality),
        "",
        text_table(units),
        "",
        "Tonnes and Bales (cotton) are never added together. Missing MSP values are left empty.",
        "",
        "## 3. Crops and production (Q1, Q2)",
        "",
        text_table(leading_crops),
        "",
        "Change in production of each state's main crop (average of the first 3 years",
        "compared with the average of the last 3 years):",
        "",
        text_table(change),
        "",
        "## 4. Descriptive statistics",
        "",
        f"Yield statistics were calculated for {len(yield_stats)} state-crop combinations",
        "(outputs/tables/yield_statistics.csv). MSP growth for the crops with the longest records:",
        "",
        text_table(msp_stats.head(8)),
        "",
        "## 5. Rainfall and yield (Q3)",
        "",
        "Rainfall is one state value per year, so district rows were first combined into one",
        "yearly total per state and crop. Yield = total production / total area.",
        "",
        text_table(correlations.head(8)[["State", "Crop", "Rainfall", "Years", "Pearson_r",
                                         "Pearson_p", "Strength"]]),
        "",
        "## 6. Seasonal yield test (Q4)",
        "",
        "Kruskal-Wallis test. H0: yield is the same across seasons. Reject H0 if p < 0.05.",
        "",
        text_table(season_tests),
        "",
        "## 7. Crop diversification (Q5)",
        "",
        "Share = crop area / state area. HHI = sum of squared shares. Diversity index = 1 - HHI.",
        "",
        text_table(latest),
        "",
        "## 8. Forecasting (Q6)",
        "",
        f"For each state the largest crop with {MIN_YEARS_FOR_FORECAST}+ years without gaps was used.",
        f"The ADF test checked stationarity; the last {TEST_YEARS} years were held back to test",
        "ARIMA(1,1,0) against a naive 'same as last year' forecast. Lower MAE is better.",
        "",
        text_table(forecasts),
        "",
        "## 9. Conclusions",
        "",
    ]

    for state, state_rows in leading_crops.groupby("State"):
        lines.append(f"- **{state}** is led by " + ", ".join(state_rows["Crop"]) + ".")
    for index, row in change.iterrows():
        lines.append(f"- {row['Crop']} production in {row['State']} changed by "
                     f"{row['Percent_Change']}% from 1997-99 to 2015-17.")
    lines += [
        f"- {len(significant)} of {len(correlations)} rainfall-yield pairs were significant "
        f"(p < 0.05). The strongest was {strongest['Crop']} in {strongest['State']} "
        f"(r = {strongest['Pearson_r']}). This is association, not proof that rain causes yield.",
        f"- In {latest['Year_Start'].iloc[0]}, {latest['State'].iloc[0]} had the most diversified "
        f"crop area and {latest['State'].iloc[-1]} the least.",
        f"- ARIMA beat the naive forecast in {arima_wins} of {len(forecasts)} states. With only about "
        "20 yearly values, forecasts are rough estimates.",
        "",
        "## 10. Limitations",
        "",
        "- Rainfall is a state-level value, not district rainfall.",
        "- MSP is a government price, not farmer profit.",
        "- Area is summed over crop-season records, so it is not unique farmland.",
        "- No data on irrigation, soil, fertiliser or input costs.",
        "",
        "Charts are in outputs/figures/. Tables are in outputs/tables/ and outputs/forecasts/.",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    return REPORT_FILE, forecasts
