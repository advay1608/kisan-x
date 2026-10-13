# KISAN-X — Agricultural Data Analysis CLI

KISAN-X is a menu-driven Python command-line project that applies the concepts in the Python Programming Laboratory syllabus to a real agricultural dataset. It examines crop production, reported cultivated area, yield, regional and temporal patterns, rainfall–yield associations, crop-area diversification, and selected crop-production forecasts.

**Design rule:** keep the application simple enough to explain in a viva. The project uses Python functions and user-defined modules, NumPy, Pandas, Matplotlib, Seaborn, `scipy.stats`, and Statsmodels time-series tools. It does not use a web framework, database, machine-learning library, or portfolio optimizer.

## Dataset

Place the final 12-column CSV at:

```text
data/this_file.csv
```

The delivered project ZIP already contains this file. Its expected columns are:

```text
Year, State, District, Crop, Season, Area_Ha, Production,
Production_Unit, Yield, Annual_Rainfall_mm, JJAS_Rainfall_mm,
MSP_Rs_per_Quintal
```

The current dataset has **102,049 observations**, covers **Maharashtra, Uttar Pradesh, Rajasthan, and Madhya Pradesh**, and spans agricultural years **1997-98 through 2017-18**. KISAN-X always prints the actual dimensions and coverage after loading the CSV instead of relying only on these documented values.

## Data source links

The dataset integrates three source families. These links identify the underlying sources rather than a new API connection; the application runs on the supplied, already-merged CSV.

- **Crop production (Government of India Open Government Data):** [District-wise, Season-wise Crop Production Statistics catalogue](https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0) and [1997-onward resource page](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997).
- **Rainfall (India Meteorological Department / OGD):** [Rainfall in India catalogue](https://data.gov.in/catalog/rainfall-india) and [Sub-Divisional Monthly Rainfall, 1901–2017 resource](https://www.data.gov.in/resource/sub-divisional-monthly-rainfall-1901-2017).
- **MSP:** the supplied `MSP_Crops_RBI_VERIFIED_v4.csv` carries the provenance for the price records. Related reference pages are the [RBI Handbook of Statistics on Indian Economy](https://rbi.org.in/Scripts/AnnualPublications.aspx) and [Government of India MSP notifications](https://desagri.gov.in/notification-type/msp-notifications/).

Rainfall is joined through a documented state-to-IMD-subdivision proxy, not a claim that meteorological subdivisions equal administrative districts. The application does not fetch updated data from these sources while running.

## Run the project

Use Python 3.10 or later.

### Windows (PowerShell) / VS Code terminal

```powershell
cd kisan-x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks environment activation, open a new terminal or use the interpreter at `.venv\Scripts\python.exe` directly.

### macOS / Linux

```bash
cd kisan-x
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The app is a standard terminal program; it does not require a server or internet connection after dependencies have been installed.

## CLI menu

1. **Dataset overview and data-quality audit** — schema, dimensions, missingness, duplicate keys, units, range checks and the relationship between yield, production and area.
2. **Exploratory analysis and charts** — state/year reported-area trends, production trends separated by unit, crop-area ranking and yield distribution.
3. **Descriptive statistics** — yield summaries grouped by state/crop/unit and MSP statistics based on distinct crop-year values.
4. **Rainfall–yield relationships** — Pearson and Spearman correlations for annual and monsoon rainfall against aggregated weighted yield. Optionally saves a scatter plot.
5. **Season comparison** — Kruskal–Wallis test of yield distributions for a selected state/crop when enough observations exist in at least two seasons.
6. **Crop diversification** — crop-area shares, HHI concentration, `1 - HHI` diversity indicator and effective number of equally sized crops.
7. **Crop-production forecasting** — select a state/crop/unit with at least 15 consecutive annual observations; compare ARIMA(1,1,0) with a naive baseline using a chronological holdout and generate a three-year forecast.
8. **Complete report** — regenerate audit and EDA outputs, correlations, crop-diversification summaries, selected season tests, charts and a forecast demonstration for each state where a suitable series exists.
0. **Exit**.

## Output locations

The program creates these folders automatically:

```text
outputs/
├── KISAN_X_ANALYSIS_REPORT.md
├── tables/
│   ├── data_audit_summary.csv
│   ├── missing_values_by_column.csv
│   ├── production_unit_coverage.csv
│   ├── area_by_state_year.csv
│   ├── production_by_state_year_unit.csv
│   ├── crop_area_by_state.csv
│   ├── yield_summary_by_state_crop.csv
│   ├── top_crops_by_state_area.csv
│   ├── production_by_crop_year.csv
│   ├── descriptive_yield_statistics.csv
│   ├── descriptive_msp_statistics.csv
│   ├── rainfall_yield_correlations.csv
│   ├── crop_area_shares.csv
│   ├── diversification_by_state_year.csv
│   ├── top_five_crop_shares_by_year.csv
│   └── hypothesis/forecast summary tables
├── figures/
│   └── PNG charts
└── forecasts/
    └── annual future forecasts and holdout evaluations
```

Outputs are reproducible and can be regenerated by selecting the relevant menu option. Running the same operation overwrites its latest output files. The source CSV is not modified.

## Methodology and guardrails

- **Unit safety:** Tonnes and Bales are not added together. Every production aggregate and forecast retains a single `Production_Unit`.
- **Area meaning:** `Area_Ha` is reported per district/crop/season record. Summing these values gives *reported crop-season area*, not necessarily unique physical land, because a field can be used in different seasons.
- **Yield:** where totals are aggregated, weighted yield is calculated as summed production divided by summed area within a single crop, state, year and production unit.
- **Missing values:** missing MSP means an MSP match is unavailable. It is not zero and is not globally imputed. The audit reports missing values; analysis removes missing pairs only where a particular calculation requires complete pairs.
- **Rainfall granularity:** rainfall is an IMD-subdivision-derived **state-level proxy**, repeated across district observations. Relationship tests first aggregate to one state/year/crop/unit record. Rainfall is not district-level weather.
- **Correlation:** Pearson and Spearman coefficients measure association, not causation. Multiple crop/state tests make the strongest results exploratory.
- **Hypothesis testing:** the season comparison uses the Kruskal–Wallis non-parametric test and reports H0, H1, statistic, p-value and the chosen alpha. The data have repeated locations/years, so the result is exploratory rather than a randomized experiment.
- **Forecasting:** only complete consecutive annual series with at least 15 observations are eligible. The final three observed years are held out chronologically. ARIMA(1,1,0) is compared with a naive previous-year baseline. The forecast is an estimate, not a guarantee; future rainfall and MSP are not model inputs.
- **No fabricated data:** missing years are never filled with zero to make a time series appear continuous.
- **No unsupported profit claims:** MSP is a benchmark price, not net farm profit. This dataset lacks full input costs, market demand, irrigation and soil measurements.

## Syllabus mapping

- **EX 2 / Python fundamentals:** menu, `if` statements, loops and input validation.
- **EX 5 / standard library:** operating-system paths and file handling through `os`.
- **EX 6 / user-defined modules:** logic is split into reusable Python files.
- **EX 7 / NumPy:** numeric arrays, vectorized arithmetic and basic numerical calculations.
- **EX 8 / Pandas:** CSV I/O, selection, missing data, grouping, descriptive summaries and sorting.
- **EX 9–10 / visualization:** Matplotlib and Seaborn line, bar, histogram and scatter charts.
- **EX 11 / SciPy:** descriptive statistics, Pearson/Spearman correlation and Kruskal–Wallis testing.
- **Time-series syllabus topic:** Statsmodels stationarity check, differencing/ARIMA and forecasting.

## Suggested implementation/demo order

For a viva demonstration, run options in this order: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**. You can explain the input data and checks before showing analytical results, then finish with forecasting and the saved report.

Read `ARCHITECTURE.md`, `docs/PROJECT_FLOW.md`, `docs/DATA_DICTIONARY.md`, and `docs/VIVA_GUIDE.md` before presenting the project.
