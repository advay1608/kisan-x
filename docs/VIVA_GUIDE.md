# KISAN-X Viva Guide

Use this as a study guide. Understand each idea and connect it to the code instead of memorizing lines.

## Project overview

**Q: What is KISAN-X?**  
A: A menu-driven Python CLI that analyzes crop production, reported area, yield, rainfall associations and crop-area diversification across four Indian states, then forecasts selected annual production series.

**Q: Why a CLI instead of a website?**  
A: The project goal is to apply our Python lab syllabus. A CLI demonstrates input handling, conditionals, loops, reusable functions and user-defined modules without adding web frameworks or databases.

**Q: What are the data sources?**  
A: Government district/season crop-production records, an IMD sub-divisional rainfall dataset mapped into state-level rainfall proxies, and the supplied verified MSP dataset. The source links and exact limitations are described in the project documentation.

## Python structure

**Q: Why did you split code into modules?**  
A: Each module has one responsibility. Loading and audit code is separate from analysis, statistics, plotting, forecasting and report generation. This improves reuse, readability and debugging.

**Q: What happens if the user enters an invalid menu choice?**  
A: The program prints a validation message and repeats the menu. It does not terminate on a normal input mistake.

**Q: Does the program change the original CSV?**  
A: No. It reads the input and writes generated outputs to the `outputs/` directory. Data conversions in memory are used for analysis and reported in the audit.

## Data and Pandas

**Q: Why do you keep production units separate?**  
A: Addition is only meaningful for compatible units. Adding Tonnes to Bales would create a meaningless total, so production summaries and forecasts are grouped by `Production_Unit`.

**Q: Why are some MSP cells missing?**  
A: The MSP table has prices for selected crops and years only. A missing match means unavailable source coverage, not a zero price. Filling these with zero would create false information.

**Q: What does the missing-value audit tell us?**  
A: It counts missing cells and reports their percentages by column. Handling then depends on the analysis: correlations use complete variable pairs; MSP summaries use existing matched price observations.

**Q: What is `groupby()` doing in this project?**  
A: It splits records into groups, applies an aggregation like `sum`, `mean` or `median`, and combines the results. For example, it groups crop production by year, state, crop and unit.

**Q: Is summed area the unique land area of a state?**  
A: Not necessarily. The input records are crop/season-specific, so multiple crop plantings on the same physical land may be counted. The project labels it reported crop-season area.

## Statistics

**Q: What is the difference between Pearson and Spearman correlation?**  
A: Pearson measures the strength/direction of a linear relationship; Spearman uses ranks and measures monotonic association. A value near +1 means a strong positive association, near -1 a strong negative association, and near 0 a weak relationship of that form.

**Q: Why isn't correlation proof of causation?**  
A: Other variables may affect both rainfall patterns and yields; this observational dataset does not isolate rainfall as the only cause.

**Q: Why aggregate rainfall before correlation?**  
A: Rainfall is a state-level value repeated on many district rows. Treating every repeated copy as independent would exaggerate the effective sample size. We aggregate to one state/year/crop/unit observation first.

**Q: What does a p-value mean here?**  
A: If the null hypothesis and test assumptions were true, the p-value measures how unusual the observed statistic or a more extreme one would be. At alpha 0.05, we reject the null when p < 0.05. It is not the probability that the null is true.

**Q: Why Kruskal–Wallis?**  
A: It is a non-parametric method taught in EX 11 for comparing distributions across independent groups. We use it to compare yield distributions across selected seasons when there are enough observations. Because the records can share years/locations, the result is explicitly exploratory.

## Diversification

**Q: What is the HHI concentration index?**  
A: Calculate each crop's share of the total reported crop-season area, square every share, then add them. A large HHI indicates that area is concentrated in a small number of crops.

**Q: What does 1 − HHI mean?**  
A: It is a simple indicator that becomes larger as crop-area shares become more evenly distributed. It is not itself a measure of profit, resilience or sustainability.

**Q: Why use area share, not production share?**  
A: Production units differ across crop records. Area is measured consistently in hectares and can be used to compare the distribution of reported crop area.

## Forecasting

**Q: Why do you need consecutive annual data?**  
A: ARIMA treats the observations as an ordered time series. Filling a missing year with zero would invent a production collapse, so incomplete series are excluded rather than fabricated.

**Q: What does ARIMA(1,1,0) mean?**  
A: The first number is the autoregressive order, the second is the differencing order, and the third is the moving-average error order. Here one difference is used to model the changes, with one lagged autoregressive term and no moving-average error term.

**Q: What is stationarity?**  
A: A stationary series has statistical properties such as mean and variance that do not systematically change over time. ADF is a test for evidence of a unit root; a p-value below 0.05 is evidence against the unit-root null, not a proof that every stationarity assumption holds.

**Q: Why compare ARIMA with a naive baseline?**  
A: A sophisticated-looking model is not automatically useful. The baseline tests whether ARIMA actually improves prediction error over simply using the previous year's production.

**Q: What do MAE and RMSE measure?**  
A: MAE is the average absolute prediction error. RMSE is the square root of the average squared error, so larger mistakes are penalized more strongly.

**Q: Why hold out the last three years?**  
A: Forecasting future years must not train on them. A chronological split approximates the real task of predicting later observations from earlier ones.

**Q: Are the forecasts guaranteed?**  
A: No. There are only 21 annual observations, and the forecast uses production history only. It does not include future rainfall, MSP, costs or market demand.

## Suggested viva demo

1. Run menu option 1 and explain the schema/missing MSP.
2. Run option 2 and explain one time trend and one crop-area chart.
3. Run option 3 and explain mean vs median and standard deviation.
4. Run option 4 and explain the rainfall aggregation level, r, p-value and why correlation is not causation.
5. Run option 5 and state H0, H1, alpha and the Kruskal–Wallis conclusion.
6. Run option 6 and calculate one crop share/HHI example by hand.
7. Run option 7 and explain the train/holdout split, ARIMA order, baseline and errors.
8. Run option 8 and point out where tables, charts, forecasts and final report were saved.
