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
module (EX 6): `main.py` is the program, `kisan_x.py` is the module it imports.

## Responsibilities

### `main.py`
Owns the menu loop, the numbered prompts, the readable error messages and the
printing of results. It loads the CSV once at startup and passes that same
DataFrame to every option. It contains no statistical formula.

### `kisan_x.py`
One module with eight numbered sections, in the same order as the menu:

| Section | Functions | Saves to |
| --- | --- | --- |
| Paths and helpers | `make_output_directories`, `save_table`, `save_figure`, `safe_name` | — |
| Loading | `load_data`, `annual_series` | — |
| 1. Audit | `audit_data` | `tables/` |
| 2. EDA | `build_eda_tables`, `make_eda_charts` | `tables/`, `figures/` |
| 3. Descriptive statistics | `descriptive_statistics` | `tables/` |
| 4. Rainfall association | `rainfall_yield_correlations`, `plot_rainfall_yield` | `tables/`, `figures/` |
| 5. Hypothesis test | `season_yield_test` | `tables/` |
| 6. Diversification | `diversification`, `plot_diversification` | `tables/`, `figures/` |
| 7. Forecasting | `forecast_candidates`, `production_series`, `forecast_production`, `error_metrics`, `year_label` | `forecasts/`, `figures/` |
| 8. Report | `markdown_table`, `generate_report` | `outputs/` |

Two functions are shared across sections:

- `load_data()` validates the 12-column schema, coerces the numeric fields in
  memory (bad values become NaN and show up in the audit) and derives
  `Year_Start` from labels such as `1997-98`. It never writes to the CSV.
- `annual_series()` collapses district and season rows to one record per
  Year-State-Crop-Unit and computes weighted yield. Every rainfall calculation
  calls it first, so repeated state rainfall values cannot be counted as
  independent observations.

## Data flow

```text
main.py
  └── kisan_x.load_data() ──────────── data/this_file.csv (read-only)

menu choice
  ├── kisan_x.audit_data() ──────────> outputs/tables/
  ├── kisan_x.build_eda_tables()
  │   + make_eda_charts() ───────────> outputs/tables/ + outputs/figures/
  ├── kisan_x.descriptive_statistics() > outputs/tables/
  ├── kisan_x.rainfall_yield_correlations() > outputs/tables/
  ├── kisan_x.season_yield_test() ───> outputs/tables/
  ├── kisan_x.diversification() ─────> outputs/tables/ + outputs/figures/
  ├── kisan_x.forecast_production() ─> outputs/forecasts/ + outputs/figures/
  └── kisan_x.generate_report() ─────> outputs/KISAN_X_ANALYSIS_REPORT.md
                                       (and every output above)
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

Do not add new libraries or modules unless a syllabus need cannot be met with the current files. The target is a complete, understandable two-file CLI—not maximum abstraction. A new analysis is a new function in `kisan_x.py` and a new entry in the `MENU` list in `main.py`, not a new file.
