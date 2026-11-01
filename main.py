"""KISAN-X menu-driven command-line application.

This file handles only the user interaction: printing the menu, reading the
choice, calling the matching function in the kisan_x module and displaying the
result. All analysis lives in kisan_x.py.
"""

import kisan_x


def show_table(table, rows=15):
    """Print the first few rows of a DataFrame in a readable way."""
    if table is None or table.empty:
        print("No results are available for this selection.")
        return
    print(table.head(rows).to_string(index=False))
    if len(table) > rows:
        print("... showing", rows, "of", len(table), "rows. The full table is in the saved CSV.")


def choose_from_list(prompt, values):
    """Show numbered options and return the chosen value.

    Accepts either the number or the name, so typing "Wheat" and "3" both work.
    """
    values = [str(value) for value in values]
    if not values:
        raise ValueError("No options are available for this step.")

    print("\nAvailable options:")
    for position, value in enumerate(values, start=1):
        print(str(position).rjust(3) + ". " + value)

    answer = input(prompt + " (name or number): ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(values):
        return values[int(answer) - 1]
    for value in values:
        if value.casefold() == answer.casefold():
            return value
    raise ValueError("Selection not recognised. Choose a listed name or number.")


def ask_state_and_crop(data):
    """Ask the user for a state, then for a crop grown in that state."""
    state = choose_from_list("Choose a state", sorted(data["State"].dropna().unique()))
    crops = sorted(data.loc[data["State"] == state, "Crop"].dropna().unique())
    crop = choose_from_list("Choose a crop", crops)
    return state, crop


# ---------------------------------------------------------------------------
# Menu options
# ---------------------------------------------------------------------------

def option_audit(data):
    audit, missing, units = kisan_x.audit_data(data)
    print("\nDATA AUDIT SUMMARY")
    show_table(audit, rows=20)
    print("\nMISSING VALUES BY COLUMN")
    show_table(missing, rows=15)
    print("\nPRODUCTION UNIT COVERAGE")
    show_table(units)
    print("\nSaved audit tables to outputs/tables/.")


def option_eda(data):
    tables = kisan_x.build_eda_tables(data)
    charts = kisan_x.make_eda_charts(data, tables)
    print("\nTOP CROPS BY REPORTED AREA (ALL YEARS COMBINED)")
    show_table(tables["top_crops"], rows=20)
    print("\nSTATE/YEAR PRODUCTION TOTALS (KEPT SEPARATE BY UNIT)")
    show_table(tables["production_by_state_year"], rows=15)
    print("\nSaved", len(charts), "figures and 4 tables.")
    print("Reminder: area is summed across crop-season records, not unique land.")


def option_descriptive(data):
    yield_stats, msp_stats = kisan_x.descriptive_statistics(data)
    print("\nYIELD STATISTICS BY STATE / CROP / UNIT")
    show_table(yield_stats, rows=20)
    print("\nMSP STATISTICS (UNIQUE CROP-YEAR PRICES)")
    show_table(msp_stats, rows=15)
    print("\nSaved both tables to outputs/tables/.")


def option_rainfall(data):
    results = kisan_x.rainfall_yield_correlations(data)
    if results.empty:
        print("No series met the minimum sample-size requirement.")
        return
    print("\nSTRONGEST ABSOLUTE PEARSON ASSOCIATIONS (EXPLORATORY)")
    show_table(results.drop(columns=["Absolute_Pearson_r"]), rows=12)
    if input("\nCreate a rainfall-yield scatter plot? (y/n): ").strip().lower() == "y":
        state, crop = ask_state_and_crop(data)
        print("Saved scatter plot:", kisan_x.plot_rainfall_yield(data, state, crop))
    print("\nAssociation is not causation; rainfall is a state-level proxy.")


def option_hypothesis(data):
    state, crop = ask_state_and_crop(data)
    result = kisan_x.season_yield_test(data, state, crop)
    print("\nKRUSKAL-WALLIS SEASON COMPARISON")
    for key, value in result.items():
        print(key.replace("_", " ") + ": " + str(value))
    print("\nThe test compares distributions; it does not show that the season caused them.")


def option_diversification(data):
    shares, metrics, top_five = kisan_x.diversification(data)
    charts = kisan_x.plot_diversification(metrics)
    latest_year = metrics["Year_Start"].max()

    print("\nDIVERSIFICATION METRICS FOR THE LATEST YEAR")
    show_table(metrics[metrics["Year_Start"] == latest_year].sort_values("State"))
    print("\nTOP FIVE CROP AREA SHARES FOR THE LATEST YEAR")
    show_table(top_five[top_five["Year_Start"] == latest_year], rows=20)
    print("\nSaved 3 tables and", len(charts), "charts.")
    print("HHI = sum of squared area shares; higher HHI means more concentration.")


def option_forecast(data):
    candidates = kisan_x.forecast_candidates(data)
    if candidates.empty:
        print("No series has enough consecutive annual observations to forecast.")
        return

    print("\nEligible series need at least", kisan_x.MIN_FORECAST_YEARS,
          "consecutive annual observations.")
    state = choose_from_list("Choose a state", sorted(candidates["State"].unique()))

    state_candidates = candidates[candidates["State"] == state]
    print("\nELIGIBLE SERIES FOR", state)
    show_table(state_candidates[["Crop", "Production_Unit", "Annual_Points",
                                 "First_Year", "Last_Year"]], rows=30)

    crop = choose_from_list("Choose a crop from the table above",
                            sorted(state_candidates["Crop"].unique()))
    units = sorted(state_candidates.loc[state_candidates["Crop"] == crop, "Production_Unit"])
    unit = units[0] if len(units) == 1 else choose_from_list("Choose a production unit", units)

    result = kisan_x.forecast_production(data, state, crop, unit)
    print("\nHOLDOUT EVALUATION (LOWER MAE AND RMSE ARE BETTER)")
    show_table(result["evaluation"][["Model", "Training_Years", "Holdout_Years",
                                     "MAE", "RMSE", "Stationarity_Note"]])
    print("\nFUTURE FORECAST")
    show_table(result["future_forecast"][["Forecast_Year", "Forecast_Production_NonNegative",
                                          "Lower_95_Percent", "Upper_95_Percent"]])
    if result["arima_mae"] < result["naive_mae"]:
        print("\nOn this holdout, ARIMA beat the naive baseline on MAE.")
    else:
        print("\nOn this holdout, ARIMA did not beat the naive baseline on MAE.")
    print("The forecast uses past production only; it is an estimate, not a guarantee.")


def option_report(data):
    print("Running the audit, EDA, statistics, correlations, tests and forecasts...")
    result = kisan_x.generate_report(data)
    print("\nREPORT CREATED:", result["report_path"])
    print("\nFORECAST SUMMARY")
    show_table(result["forecast_summary"], rows=12)
    print("\nAll supporting tables and figures are in the outputs folder.")


MENU = [
    ("1", "Dataset overview and data-quality audit", option_audit),
    ("2", "Exploratory analysis and charts", option_eda),
    ("3", "Descriptive statistical summaries", option_descriptive),
    ("4", "Rainfall and yield relationships", option_rainfall),
    ("5", "Compare yield across seasons (hypothesis test)", option_hypothesis),
    ("6", "Crop diversification analysis", option_diversification),
    ("7", "Forecast crop production", option_forecast),
    ("8", "Generate the complete analysis report", option_report),
]


def main():
    print("\n" + "=" * 62)
    print("                         KISAN-X")
    print("       CROP PRODUCTION ANALYSIS AND FORECASTING SYSTEM")
    print("=" * 62)

    kisan_x.make_output_directories()
    try:
        data = kisan_x.load_data()
    except (FileNotFoundError, ValueError, OSError) as error:
        print("\nUnable to load the dataset:")
        print(error)
        return

    print("Loaded", len(data), "rows from data/this_file.csv (read-only).")
    actions = {key: function for key, _, function in MENU}

    while True:
        print("\nMAIN MENU")
        for key, label, _ in MENU:
            print(key + ". " + label)
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()
        if choice == "0":
            print("Thank you for using KISAN-X. Goodbye!")
            break
        if choice not in actions:
            print("Invalid choice. Enter a number from 0 to 8.")
            continue

        try:
            actions[choice](data)
        except (ValueError, FileNotFoundError, OSError) as error:
            print("\nOperation could not be completed:", error)
        except Exception as error:
            # Unexpected problems are shown rather than hidden, so a wrong
            # number is never mistaken for a correct result.
            print("\nUnexpected error:", type(error).__name__, "-", error)


if __name__ == "__main__":
    main()
