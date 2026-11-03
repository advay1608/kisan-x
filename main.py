"""KISAN-X: crop production analysis and forecasting. Run this file to start."""

import kisan_x


# ---------------- Helpers ----------------

def show_table(table, rows=15):
    if table is None or table.empty:
        print("No results to show.")
        return
    print(table.head(rows).to_string(index=False))
    if len(table) > rows:
        print(f"... showing {rows} of {len(table)} rows (full table saved as CSV)")


def choose_from_list(prompt, options):
    options = list(options)
    print()
    for number, option in enumerate(options, start=1):
        print(f"{number:>3}. {option}")

    answer = input(f"{prompt} (number or name): ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    for option in options:
        if option.lower() == answer.lower():
            return option
    raise ValueError("That choice is not in the list.")


def choose_state_and_crop(data):
    state = choose_from_list("Choose a state", sorted(data["State"].unique()))
    crops = sorted(data.loc[data["State"] == state, "Crop"].unique())
    crop = choose_from_list("Choose a crop", crops)
    return state, crop


# ---------------- Menu options ----------------

def option_data_quality(data):
    quality, missing, units = kisan_x.check_data_quality(data)
    print("\nDATA QUALITY CHECKS")
    show_table(quality)
    print("\nMISSING VALUES")
    show_table(missing)
    print("\nPRODUCTION UNITS")
    show_table(units)


def option_explore(data):
    top_crops, production, charts = kisan_x.explore_data(data)
    print("\nTOP CROPS BY AREA IN EACH STATE")
    show_table(top_crops, rows=40)
    print(f"\n{len(charts)} charts saved in outputs/figures/")


def option_statistics(data):
    yield_stats, msp_stats, msp_chart = kisan_x.describe_data(data)
    print("\nYIELD STATISTICS (per state and crop)")
    show_table(yield_stats)
    print("\nMSP STATISTICS")
    show_table(msp_stats)
    print("\nMSP chart saved:", msp_chart)


def option_rainfall(data):
    results = kisan_x.rainfall_yield_correlation(data)
    print("\nSTRONGEST RAINFALL-YIELD CORRELATIONS")
    show_table(results, rows=10)
    print("Correlation shows association only, not cause and effect.")

    if input("\nDraw a rainfall vs yield chart? (y/n): ").strip().lower() == "y":
        state, crop = choose_state_and_crop(data)
        print("Chart saved:", kisan_x.plot_rainfall_yield(data, state, crop))


def option_seasons(data):
    state, crop = choose_state_and_crop(data)
    result = kisan_x.compare_seasons(data, state, crop)
    print("\nKRUSKAL-WALLIS TEST")
    print("H0: yield is the same in every season")
    for key, value in result.items():
        print(f"{key}: {value}")


def option_diversity(data):
    diversity, chart = kisan_x.crop_diversity(data)
    latest = diversity[diversity["Year_Start"] == diversity["Year_Start"].max()]
    print("\nCROP DIVERSITY IN THE LATEST YEAR")
    show_table(latest.sort_values("Diversity_Index", ascending=False))
    print("Higher diversity index = area spread more evenly across crops.")
    print("Chart saved:", chart)


def option_forecast(data):
    series = kisan_x.forecastable_series(data)
    state = choose_from_list("Choose a state", sorted(series["State"].unique()))
    state_series = series[series["State"] == state]
    show_table(state_series, rows=40)

    crop = choose_from_list("Choose a crop", list(state_series["Crop"].unique()))
    units = list(state_series.loc[state_series["Crop"] == crop, "Production_Unit"])
    unit = units[0]
    if len(units) > 1:
        unit = choose_from_list("Choose a unit", units)

    result = kisan_x.forecast_production(data, state, crop, unit)
    print("\nStationarity:", result["stationarity"])
    print("\nTEST ON THE LAST YEARS (lower error is better)")
    show_table(result["evaluation"])
    print("Better model:", result["better_model"])
    print("\nFUTURE FORECAST")
    show_table(result["forecast"])
    print("Chart saved:", result["chart"])


def option_report(data):
    print("Running every analysis, please wait...")
    report_file, forecasts = kisan_x.generate_report(data)
    print("\nFORECAST SUMMARY")
    show_table(forecasts)
    print("\nReport saved:", report_file)


# ---------------- Main menu ----------------

def print_menu():
    print("\n========== KISAN-X MENU ==========")
    print("1. Data quality check")
    print("2. Explore crops, area and production")
    print("3. Descriptive statistics (yield and MSP)")
    print("4. Rainfall vs yield")
    print("5. Compare yield across seasons")
    print("6. Crop diversification")
    print("7. Forecast production")
    print("8. Generate full report")
    print("0. Exit")


def main():
    print("KISAN-X - Crop Production Analysis and Forecasting")
    kisan_x.make_output_folders()
    try:
        data = kisan_x.load_data()
    except (FileNotFoundError, ValueError) as error:
        print("Could not load the data:", error)
        return
    print(f"Loaded {len(data)} rows from data/this_file.csv")

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        try:
            if choice == "1":
                option_data_quality(data)
            elif choice == "2":
                option_explore(data)
            elif choice == "3":
                option_statistics(data)
            elif choice == "4":
                option_rainfall(data)
            elif choice == "5":
                option_seasons(data)
            elif choice == "6":
                option_diversity(data)
            elif choice == "7":
                option_forecast(data)
            elif choice == "8":
                option_report(data)
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Please enter a number from 0 to 8.")
        except ValueError as error:
            print("Error:", error)
        except Exception as error:
            print("Something went wrong:", error)


if __name__ == "__main__":
    main()
