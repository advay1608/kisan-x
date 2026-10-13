# KISAN-X Dataset Dictionary

The input file is `data/this_file.csv`. The project expects these twelve columns.

| Column | Meaning | Unit / type | Use and caution |
|---|---|---|---|
| `Year` | Agricultural-year label, such as `1997-98` | Text | Kept as a label; an integer starting year is derived in memory only to sort and process time series. |
| `State` | State represented by the source crop record | Category | Current submission includes Maharashtra, Uttar Pradesh, Rajasthan and Madhya Pradesh. |
| `District` | District represented by the crop record | Category | Crop observations are at district level; rainfall is not. |
| `Crop` | Source crop label | Category | Labels are not silently merged or renamed. |
| `Season` | Crop season, for example Kharif, Rabi, Whole Year, Autumn or Summer | Category | The same crop may be reported in more than one season. |
| `Area_Ha` | Reported area for the district/crop/season/year observation | Hectares | Sums are called *reported crop-season area*. They may count multiple crop/season plantings, not unique physical land. |
| `Production` | Reported crop production | Number in the associated `Production_Unit` | Aggregate only within the same unit. Zero is retained as a valid source value. |
| `Production_Unit` | Unit attached to production | Category, currently including Tonnes and Bales | Different units are not added together. Do not convert Bales to Tonnes without crop-specific documented factors. |
| `Yield` | Production per hectare as provided by the source | Production unit per hectare | This dataset's values agree with `Production / Area_Ha` within floating-point tolerance. Yield comparisons should be crop- and unit-aware. |
| `Annual_Rainfall_mm` | Annual rainfall proxy joined to the state/year | Millimetres | Derived from IMD meteorological subdivisions. It is state-level, not district-level. |
| `JJAS_Rainfall_mm` | June–September rainfall proxy | Millimetres | Common monsoon-period rainfall measure. Same state/year value repeats across multiple crop records. |
| `MSP_Rs_per_Quintal` | Matched Minimum Support Price | Rupees per quintal | Missing values indicate no matched MSP value in the supplied MSP dataset; they do not mean zero price. MSP is not net profit. |

## Important data semantics

1. The original CSV is preserved. The application converts numeric columns to numeric types in memory only, and it records values that cannot be parsed as missing in the audit.
2. Missing MSP is expected because the MSP source covers selected crops/years only.
3. Rainfall covers the common 1997–2017 calendar-year-start window and is joined using the first year of the agricultural-year label. It is a state-level proxy derived from IMD sub-divisions.
4. The rainfall columns repeat for records in the same state/year. For an annual crop relationship, aggregate to one `State + Year + Crop + Production_Unit` record first.
5. A production total across a state's crops can be reported per unit, such as total tonnes; it must not add Bales to Tonnes.
6. A grouped weighted yield is `sum(Production) / sum(Area_Ha)` for the same state, year, crop and production unit.
