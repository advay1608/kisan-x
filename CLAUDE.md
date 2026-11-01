# CLAUDE.md — KISAN-X Project Rules

## Mission
Build KISAN-X as a clear, reproducible, menu-driven Python CLI for agricultural data analysis. It must address the stated problem: production, cultivated area, yield, regional/temporal patterns, statistical relationships, diversification and selected production forecasts.

## Hard constraints
1. Keep exactly two Python source files: `main.py` (menu and user interaction) plus `kisan_x.py` (the user-defined analysis module). Do not add more files or a `src/` package.
2. Use only core Python/standard library, NumPy, Pandas, Matplotlib, Seaborn, `scipy.stats`, and Statsmodels time-series tools.
3. Do not add Streamlit, Flask, FastAPI, Django, Plotly, scikit-learn, TensorFlow, PyTorch, databases, APIs, portfolio optimizers or other third-party frameworks.
4. The input file is `data/this_file.csv`. Never overwrite or edit it.
5. The required schema is the 12 columns documented in `docs/DATA_DICTIONARY.md`. If the columns differ, report the mismatch rather than guessing.
6. Never apply global `dropna()`, silently delete duplicate rows, or fill missing MSP/rainfall with zero or the mean.
7. Never sum production across different `Production_Unit` values. No unsubstantiated Tonnes/Bales conversion.
8. Do not describe summed crop-season area as unique physical land area.
9. Rainfall values are state-level IMD proxies repeated across district records. Before rainfall correlation, aggregate to state/year/crop/unit and calculate weighted yield; never count each district copy as an independent rainfall observation.
10. Correlation is association, not causation. MSP is not profit; no complete input-cost, irrigation, soil or market-demand variables are available.
11. Forecast only sufficiently long consecutive annual series. Never fill missing years with zero. Use chronological holdout evaluation, compare a naive baseline with ARIMA(1,1,0), and report uncertainty/limitations.
12. Keep code beginner-friendly: functions, clear names, comments that explain analytical assumptions, limited abstraction and readable errors.
13. Every major operation must save its table to `outputs/tables/`, charts to `outputs/figures/`, forecasts/evaluations to `outputs/forecasts/`, or the report to `outputs/`.
14. Do not add class hierarchies, design patterns, automatic model search or features the student cannot explain in a viva.
15. Keep the two files readable: `main.py` holds only menu, prompts and printing; `kisan_x.py` holds only analysis functions. Analysis logic never goes in `main.py`, and `input()`/menus never go in `kisan_x.py`.

## Syllabus coverage
- EX 2: menu, branches, loops and input handling.
- EX 5: `os` paths/directories and file access.
- EX 6: user-defined modules and reusable functions.
- EX 7: NumPy arrays and vectorized numeric calculations.
- EX 8: Pandas CSV I/O, filters, grouping, sorting, missing-value summaries.
- EX 9–10: Matplotlib/Seaborn line, bar, histogram and scatter plots.
- EX 11: SciPy descriptive statistics, correlation and Kruskal–Wallis testing.
- Syllabus time-series topics: stationarity checking, smoothing/differencing, ARIMA and forecast evaluation in Statsmodels.

## Required workflow
1. Verify data path/schema.
2. Run audit and inspect missingness/units.
3. Generate EDA tables and charts.
4. Generate descriptive statistics.
5. Analyze rainfall-yield association after correct aggregation.
6. Run selected hypothesis tests and crop-area diversification.
7. Fit/evaluate forecasts for complete annual series.
8. Generate and inspect final report.

Before adding a feature ask: Is it explicitly in the PS? Is the dataset capable of supporting it? Is it inside the taught syllabus? Can the student explain the formula, output and limitation? If not, do not add it.
