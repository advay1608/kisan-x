# KISAN-X Architecture

## Design goals

1. Implement every component of the stated problem: production, area, yield, regional/temporal patterns, statistical relationships, crop diversification and selected production forecasts.
2. Stay inside the student's lab syllabus and keep modules understandable in a viva.
3. Keep the delivered source dataset unchanged.
4. Make every major analysis save its underlying table or chart for verification.
5. Avoid scientifically misleading aggregation caused by inconsistent production units or repeated state-level rainfall values.

## File structure

```text
kisan-x/
├── main.py                          # Menu, prompts and printing only
├── kisan_x.py                       # User-defined module: all analysis functions
├── requirements.txt                 # Only the taught third-party libraries
├── README.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── data/
│   └── this_file.csv                # Source dataset (read-only input)
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── PROJECT_FLOW.md
│   └── VIVA_GUIDE.md
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── forecasts/
└── logs/
```

There are **two Python source files total**. No package, database, web server or
dashboard is used. The split follows the syllabus requirement for a user-defined
module (syllabus unit 2.2): `main.py` is the program, `kisan_x.py` is the module it imports.

## Responsibilities

### `main.py`
Owns the menu loop, the numbered prompts, the readable error messages and the
printing of results. It loads the CSV once at startup and passes that same
DataFrame to every option. It contains no statistical formula.

### `kisan_x.py`
One module with eight numbered sections, in the same order as the menu:

| Section | Functions | Saves to |
| --- | --- | --- |
| Saving files | `make_output_folders`, `save_table`, `save_chart`, `clean_name`, `line_chart_by_state` | — |
| Loading | `load_data`, `yearly_totals` | — |
| 1. Data quality | `find_yield_outliers`, `check_data_quality` | `tables/` |
| 2. Exploration | `explore_data` | `tables/`, `figures/` |
| 3. Descriptive statistics | `describe_data` | `tables/`, `figures/` |
| 4. Rainfall and yield | `correlation_strength`, `rainfall_yield_correlation`, `plot_rainfall_yield` | `tables/`, `figures/` |
| 5. Season test | `compare_seasons` | `tables/`, `figures/` |
| 6. Diversification | `crop_diversity` | `tables/`, `figures/` |
| 7. Forecasting | `forecastable_series`, `year_label`, `mae_and_rmse`, `fit_arima`, `forecast_production` | `forecasts/`, `figures/` |
| 8. Report | `season_tests_for_report`, `forecasts_for_report`, `production_change`, `text_table`, `generate_report` | `outputs/` |

Two functions are shared across sections:

- `load_data()` checks the 12 columns, converts the numeric fields in memory
  (bad values become NaN) and makes `Year_Start` from labels such as `1997-98`.
  It never writes to the CSV.
- `yearly_totals()` combines district and season rows into one record per
  State-Crop-Unit-Year and computes weighted yield. Rainfall is taken once per
  state-year (`"first"`), so repeated copies are not counted as extra data.

## Data flow

```text
main.py
  └── kisan_x.load_data() ──────────────── data/this_file.csv (read-only)

menu choice
  1 ── check_data_quality() ────────────> outputs/tables/
  2 ── explore_data() ──────────────────> outputs/tables/ + outputs/figures/
  3 ── describe_data() ─────────────────> outputs/tables/ + outputs/figures/
  4 ── rainfall_yield_correlation() ────> outputs/tables/ (+ plot_rainfall_yield)
  5 ── compare_seasons() ───────────────> outputs/tables/ + outputs/figures/
  6 ── crop_diversity() ────────────────> outputs/tables/ + outputs/figures/
  7 ── forecast_production() ───────────> outputs/forecasts/ + outputs/figures/
  8 ── generate_report() ───────────────> outputs/KISAN_X_ANALYSIS_REPORT.md
```

## Statistical design rules

- State-level rainfall has one value per state/year in the data. Correlation analysis reduces district/crop-season records to annual state/crop/unit records before calculating Pearson and Spearman values.
- Crop production sums only across rows with the same unit. No Tonnes/Bales conversion is attempted.
- Annual weighted yield for an aggregated crop series is `sum(Production) / sum(Area_Ha)` inside one state, crop, year and production unit.
- MSP rows are repeated on the joined district records. Summary statistics deduplicate crop/year/MSP combinations only within the MSP summary calculation; the input data remain unchanged.
- Area totals are described as sums of reported crop-season area, not unique physical land.
- Forecast validation is time-ordered, not randomly shuffled. The naive baseline helps determine whether ARIMA improves on a simple rule.
- Tests report sample sizes, hypotheses, statistic, p-value and alpha. Statistical significance is not proof of causation.

## Error handling

Expected problems such as missing files, missing schema columns, invalid menu selections or insufficient time-series observations raise readable exceptions and return to the menu. An unexpected exception is displayed instead of silently ignored.

## Extension policy

Do not add new libraries or modules unless a syllabus need cannot be met with the current files. The target is a complete, understandable two-file CLI—not maximum abstraction. A new analysis is a new function in `kisan_x.py` and a new line in `print_menu()` plus a new `elif` branch in `main()`, not a new file.
