"""KISAN-X menu-driven command-line program.

This file only talks to the user: it shows the menu, reads choices, calls the
matching function in kisan_x.py and prints the results. All analysis is in
kisan_x.py (our user-defined module).
"""

import kisan_x


# ===========================================================================
# Input and printing helpers
# ===========================================================================

def show_table(table, rows=15):
    """Print the first few rows of a table."""
    if table is None or table.empty:
        print("No results are available for this selection.")
        return
    print(table.head(rows).to_string(index=False))
    if len(table) > rows:
        print(f"... showing {rows} of {len(table)} rows. The full table is in the saved CSV.")


def choose_from_list(prompt, options):
    """Print numbered options and return the one the user picks.

    The user can type the number (e.g. 3) or the name (e.g. Wheat).
    """
    options = [str(option) for option in options]
    if not options:
        raise ValueError("No options are available for this step.")

    print("\nAvailable options:")
    for number, option in enumerate(options, start=1):
        print(f"{number:>3}. {option}")

    answer = input(f"{prompt} (name or number): ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    for option in options:
        if option.lower() == answer.lower():
            return option
    raise ValueError("Selection not recognised. Choose a listed name or number.")


def ask_state_and_crop(data):
    """Ask for a state, then for a crop grown in that state."""
    state = choose_from_list("Choose a state", sorted(data["State"].dropna().unique()))
    crops_in_state = sorted(data.loc[data["State"] == state, "Crop"].dropna().unique())
    crop = choose_from_list("Choose a crop", crops_in_state)
    return state, crop


# ===========================================================================
# One function per menu option
# ===========================================================================

def option_audit(data):
    audit, missing, units = kisan_x.audit_data(data)
    print("\nDATA AUDIT SUMMARY")
    show_table(audit, rows=20)
    print("\nMISSING VALUES BY COLUMN")
    show_table(missing)
    print("\nPRODUCTION UNIT COVERAGE")
    show_table(units)
    print("\nSaved audit tables to outputs/tables/.")


def option_eda(data):
    tables = kisan_x.build_eda_tables(data)
    charts = kisan_x.make_eda_charts(data, tables)
    print("\nTOP CROPS BY REPORTED AREA (ALL YEARS COMBINED)")
    show_table(tables["top_crops"], rows=20)
    print("\nSTATE/YEAR PRODUCTION TOTALS (KEPT SEPARATE BY UNIT)")
    show_table(tables["production_by_state_year"])
    print(f"\nSaved {len(charts)} figures and {len(tables)} tables.")
    print("Reminder: area is summed across crop-season records, not unique land.")


def option_descriptive(data):
    yield_stats, msp_stats = kisan_x.descriptive_statistics(data)
    print("\nYIELD STATISTICS BY STATE / CROP / UNIT")
    show_table(yield_stats, rows=20)
    print("\nMSP STATISTICS (UNIQUE CROP-YEAR PRICES)")
    show_table(msp_stats)
    print("\nSaved both tables to outputs/tables/.")


def option_rainfall(data):
    results = kisan_x.rainfall_yield_correlations(data)
    if results.empty:
        print("No series met the minimum sample-size requirement.")
        return
    print("\nSTRONGEST ABSOLUTE PEARSON ASSOCIATIONS (EXPLORATORY)")
    show_table(results.drop(columns=["Absolute_Pearson_r"]), rows=12)

    answer = input("\nCreate a rainfall-yield scatter plot? (y/n): ").strip().lower()
    if answer == "y":
        state, crop = ask_state_and_crop(data)
        print("Saved scatter plot:", kisan_x.plot_rainfall_yield(data, state, crop))
    print("\nAssociation is not causation; rainfall is a state-level proxy.")


def option_hypothesis(data):
    state, crop = ask_state_and_crop(data)
    result = kisan_x.season_yield_test(data, state, crop)
    print("\nKRUSKAL-WALLIS SEASON COMPARISON")
    for key, value in result.items():
        print(f"{key.replace('_', ' ')}: {value}")
    print("\nThe test compares distributions; it does not show that the season caused them.")


def option_diversification(data):
    shares, metrics, top_five = kisan_x.diversification(data)
    charts = kisan_x.plot_diversification(metrics)
    latest_year = metrics["Year_Start"].max()

    print("\nDIVERSIFICATION METRICS FOR THE LATEST YEAR")
    show_table(metrics[metrics["Year_Start"] == latest_year].sort_values("State"))
    print("\nTOP FIVE CROP AREA SHARES FOR THE LATEST YEAR")
    show_table(top_five[top_five["Year_Start"] == latest_year], rows=20)
    print(f"\nSaved 3 tables and {len(charts)} charts.")
    print("HHI = sum of squared area shares; higher HHI means more concentration.")


def option_forecast(data):
    candidates = kisan_x.forecast_candidates(data)
    if candidates.empty:
        print("No series has enough consecutive annual observations to forecast.")
        return

    print(f"\nEligible series need at least {kisan_x.MIN_FORECAST_YEARS} "
          "consecutive annual observations.")
    state = choose_from_list("Choose a state", sorted(candidates["State"].unique()))

    in_state = candidates[candidates["State"] == state]
    print("\nELIGIBLE SERIES FOR", state)
    show_table(in_state[["Crop", "Production_Unit", "Annual_Points", "First_Year", "Last_Year"]],
               rows=30)

    crop = choose_from_list("Choose a crop from the table above", sorted(in_state["Crop"].unique()))
    units = sorted(in_state.loc[in_state["Crop"] == crop, "Production_Unit"])
    if len(units) == 1:
        unit = units[0]
    else:
        unit = choose_from_list("Choose a production unit", units)

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


# ===========================================================================
# Main menu loop
# ===========================================================================

def print_menu():
    print("\nMAIN MENU")
    print("1. Dataset overview and data-quality audit")
    print("2. Exploratory analysis and charts")
    print("3. Descriptive statistical summaries")
    print("4. Rainfall and yield relationships")
    print("5. Compare yield across seasons (hypothesis test)")
    print("6. Crop diversification analysis")
    print("7. Forecast crop production")
    print("8. Generate the complete analysis report")
    print("0. Exit")


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
    print(f"Loaded {len(data)} rows from data/this_file.csv (read-only).")

    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()

        try:
            if choice == "1":
                option_audit(data)
            elif choice == "2":
                option_eda(data)
            elif choice == "3":
                option_descriptive(data)
            elif choice == "4":
                option_rainfall(data)
            elif choice == "5":
                option_hypothesis(data)
            elif choice == "6":
                option_diversification(data)
            elif choice == "7":
                option_forecast(data)
            elif choice == "8":
                option_report(data)
            elif choice == "0":
                print("Thank you for using KISAN-X. Goodbye!")
                break
            else:
                print("Invalid choice. Enter a number from 0 to 8.")
        except (ValueError, OSError) as error:
            # Expected problems: a wrong selection, too little data, or a file error.
            print("\nOperation could not be completed:", error)
        except Exception as error:
            # Anything unexpected is shown, never hidden, so a wrong result
            # is never mistaken for a correct one.
            print("\nUnexpected error:", type(error).__name__, "-", error)


if __name__ == "__main__":
    main()
