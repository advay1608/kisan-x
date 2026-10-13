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
├── main.py                         # CLI entry point and menu
├── requirements.txt                # Only the taught third-party libraries
├── README.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── data/
│   └── this_file.csv                # User/source dataset (read-only input)
├── src/
│   ├── data_utils.py                # Loader, schema check, audit, output folders
│   ├── analysis.py                  # EDA summary tables and temporal aggregations
│   ├── statistical_analysis.py      # Descriptive stats, correlations, hypothesis test
│   ├── diversification.py            # Crop shares, HHI and diversity summary
│   ├── forecasting.py               # Consecutive series, ARIMA and evaluation
│   ├── visualization.py             # PNG charts
│   └── report.py                    # Orchestrates a reproducible analysis report
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

There are **eight Python source files total**: `main.py` plus seven modules under `src/`. No separate application package, database, web server, or dashboard is used.

## Responsibilities and interfaces

### `main.py`
Owns the menu, prompts, readable error messages and top-level function calls. It reads the file once at startup and keeps the original loaded DataFrame for each operation. It does not implement statistical formulas directly.

### `src/data_utils.py`
- Defines the required schema.
- Loads CSV through `pandas.read_csv`.
- Coerces numeric analytical fields to numbers in memory; invalid values become missing and are visible in the audit.
- Reports schema, missingness, duplicates, unit coverage, negative values, `Yield` versus `Production / Area`, and rainfall uniqueness at State-Year level.
- Creates output directories.
- Never writes or overwrites `data/this_file.csv`.

### `src/analysis.py`
Creates summary tables: reported crop area by state/year, production by state/year/unit, crop area by state/crop, yield summaries and crop-production time series. Production aggregation always includes `Production_Unit`.

### `src/statistical_analysis.py`
Creates grouped descriptive statistics, summaries of distinct crop-year MSP values, rainfall-yield correlations and the selected Kruskal–Wallis test. It aggregates records before inference to prevent multiple copies of state-level rainfall from being treated as independent weather observations.

### `src/diversification.py`
Computes crop-area shares by state/year and simple crop-concentration/diversity measures. Formulae are documented in the output and project flow docs.

### `src/forecasting.py`
Identifies complete annual series, aggregates one state/crop/unit at a time, evaluates ARIMA(1,1,0) on the final three observed years, compares it with a naive previous-year baseline, refits on the available history and generates three future points. No missing year is silently filled.

### `src/visualization.py`
Creates titled, labelled and saved Matplotlib/Seaborn charts. The backend is non-interactive so the CLI can save charts without requiring a GUI window.

### `src/report.py`
Runs audit, EDA, statistics, correlations, diversification, plots, one eligible season-test attempt per state and one explainably selected forecasting series per state. It writes `outputs/KISAN_X_ANALYSIS_REPORT.md` and supporting files.

## Data flow

```text
main.py
  └── data_utils.load_dataset()
        └── data/this_file.csv (read-only)

CLI choice
  ├── data_utils.run_data_audit() ───────> outputs/tables/
  ├── analysis.build_eda_tables() ───────> outputs/tables/
  ├── statistical_analysis.*() ─────────> outputs/tables/
  ├── diversification.*() ───────────────> outputs/tables/
  ├── forecasting.*() ──────────────────> outputs/forecasts/ + figure
  ├── visualization.*() ────────────────> outputs/figures/
  └── report.generate_full_report() ─────> Markdown report + all outputs
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

Do not add new libraries or modules unless a syllabus need cannot be met with the current files. The target is a complete, understandable eight-file CLI—not maximum abstraction.
