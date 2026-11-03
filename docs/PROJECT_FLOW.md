# KISAN-X Project Flow and Analysis Plan

## Problem statement

Analyze historical crop production, cultivated area, yield and related agricultural factors across selected Indian states and districts; identify regional and temporal patterns; examine statistical relationships and crop diversification; and forecast selected crop-production trends for data-driven agricultural insights.

## End-to-end flow

```text
Input CSV (data/this_file.csv)
             |
             v
     1. Load + schema check
             |
             v
     2. Data-quality audit
             |
             v
     3. EDA summary tables + plots
             |
      +------+------+-------------------+
      |             |                   |
      v             v                   v
  Statistics   Rainfall/yield       Crop-area
  and tests    relationships       diversification
      |             |                   |
      +-------------+-------------------+
                    |
                    v
          4. Forecast selected series
             (ARIMA vs naive baseline)
                    |
                    v
          5. Final Markdown report
             + CSV tables + PNG figures
```

## Phase 1 — Load and audit

**Question:** What is in the dataset, and can the code safely analyze it?

The audit checks the required columns, row/column count, number of states/districts/crops/seasons/units, missingness by column, duplicate rows, duplicate analytical keys, negative numeric values, and consistency between `Yield` and `Production / Area_Ha`. It also checks that each state/year has one distinct rainfall proxy value.

Missing MSP is not treated as an error or imputed to zero. The CSV is never modified or automatically stripped of rows.

## Phase 2 — Explore production, area and yield

**Questions:** Which crops have the most reported area? How do state/year totals change? What do yield distributions look like?

Output tables:
- `area_by_state_year.csv`: sums the reported area across district/crop/season rows by state/year.
- `production_by_state_year_unit.csv`: production trends by state/year/unit; units remain separate.
- `crop_area_by_state.csv` and `top_crops_by_state_area.csv`: crop-area ranking.
- `yield_summary_by_state_crop.csv`: count, mean, median, standard deviation, minimum and maximum by state/crop/unit.
- `production_by_crop_year.csv`: annual production and area for each state/crop/unit.

Charts include area trends, production trends by unit, top crop-area bars, yield histogram, an MSP trend line, rainfall-yield regression scatter plots and season box plots.

## Phase 3 — Descriptive statistics

For each state/crop/unit yield series, the program computes record count, mean, median, standard deviation, minimum, maximum, skewness and coefficient of variation (CV = std / mean x 100). These summaries can reveal spread, asymmetry and extreme values, but they do not explain why a crop performed differently.

MSP statistics use distinct crop/year/MSP values so that a single crop-year price repeated across many district observations is not counted as a new price for every district.

## Phase 4 — Rainfall and yield relationships

**Question:** Is yearly rainfall associated with crop yield for a selected state/crop?

Procedure:
1. Group district/crop-season observations to a single `State + Year + Crop + Production_Unit` annual record.
2. Sum area and production within that compatible group.
3. Calculate weighted yield: `total production / total area`.
4. Take the state/year rainfall value once, instead of treating district duplicates as independent rainfall records.
5. Calculate Pearson correlation (linear association) and Spearman correlation (rank/monotonic association) between yield and annual or JJAS rainfall.
6. Save correlation coefficients, p-values and sample sizes.

Interpretation: a positive coefficient means higher rainfall tended to accompany higher yield in that series; a negative coefficient means the opposite pattern. Neither proves that rainfall caused the yield changes. Other influences may include soil, irrigation, crop variety, prices and technology. Because many series are explored, the strongest result is exploratory unless confirmed independently.

## Phase 5 — Hypothesis testing

The CLI lets the user choose a state and crop and compares yield distributions across seasons with the **Kruskal–Wallis H-test**, a non-parametric test covered by the SciPy assignment.

- **H0:** the selected season yield distributions are the same.
- **H1:** at least one selected season has a different yield distribution.
- **Alpha:** 0.05.
- A p-value below 0.05 is interpreted as evidence against H0 for this comparison; otherwise there is insufficient evidence to reject H0.

At least two seasons must each contain at least five yield observations. The test is exploratory: rows may share locations and years, and the dataset is not a randomized experiment.

## Phase 6 — Crop diversification

For each state/year:
1. Sum reported crop-season area for each crop.
2. Calculate each crop's area share `s_i = crop area / total reported crop-season area`.
3. Calculate the Herfindahl concentration index `HHI = sum(s_i ** 2)`.
4. Calculate `Diversity_Index = 1 - HHI`.

HHI is near 1 when one crop dominates reported area; it is smaller when area is divided among many crops. `1 - HHI` increases with a more even area distribution. These are descriptive indices, not profitability or sustainability scores.

## Phase 7 — Forecasting

**Question:** What does a simple time-series model estimate for selected future production years, and does it outperform a basic baseline on recent held-out data?

Procedure:
1. Choose a state, crop and production unit.
2. Sum district/season production by year for only that crop and unit.
3. Require at least 15 consecutive annual observations. Missing years are not filled with zero.
4. Hold out the last three years in chronological order; train only on earlier years.
5. Run an Augmented Dickey–Fuller (ADF) stationarity test on the training series as an educational diagnostic.
6. Fit Statsmodels `ARIMA(1,1,0)` to the training series and predict the holdout period.
7. Compare it with a naive previous-year baseline using:
   - `MAE = mean(abs(actual - predicted))`
   - `RMSE = sqrt(mean((actual - predicted) ** 2))`
8. Refit ARIMA on the full observed history and forecast three future annual values with 95% model intervals.

A lower MAE/RMSE is better on the holdout, but one three-year holdout is weak evidence of general reliability. ARIMA(1,1,0) is deliberately fixed so the model stays understandable. Future rainfall and MSP are not inputs, so this is a univariate production forecast—not a weather-impact model. Negative raw forecasts are shown alongside a non-negative presentation value because production cannot be negative; the raw prediction is preserved for transparency.

## Phase 8 — Final report

Menu option 8 regenerates supporting tables and charts and writes `outputs/KISAN_X_ANALYSIS_REPORT.md`. It selects one complete high-area crop series per state for a forecast demonstration and attempts a season test for each state's largest cumulative reported-area crop when enough season groups exist. The selection rule and any failed/skipped analysis are written into outputs.

## Scope boundaries

This is an educational decision-support analysis, not a farm advisory model. Data do not contain full production costs, soil composition, irrigation levels, crop variety, market demand or daily weather. Do not claim causal effects, profit, or guaranteed forecasts.
