# Notes on Housing Data Cleaning and Merging Process

### Overview of the Process

The goal was to clean, merge, and analyze two housing datasets to find common properties and remove any discrepancies. The steps involved included cleaning, transforming, and merging the datasets, followed by a quality control analysis. Below are detailed notes capturing each stage of the workflow.

### Files Used in the Process

1. **Input Files**:
   - `Housing Prices Dataset.csv`: Original dataset containing housing price information.
   - `houseprice.csv`: Second dataset with potentially different or additional housing data.

2. **Intermediate File**:
   - `cleaned_merged_housing_dataset.csv`: Result of initial cleaning and merging, used for verification.

3. **Output Files**:
   - `merged_housing_dataset.csv`: Cleaned and merged dataset.
   - `further_cleaned_standardized_housing_dataset.csv`: Final dataset after additional cleaning and standardization.

4. **Documentation**:
   - `Data preparation.md`: This file, documenting the entire data preparation process.


### Steps Taken

#### 1. Loading the Datasets
- **Files Loaded**:
  - `Housing Prices Dataset.csv`
  - `houseprice.csv`
  - `cleaned_merged_housing_dataset.csv` (loaded with a verification check to avoid empty file errors)
- Used a custom function to handle empty files and provide warnings in case any file was empty or missing.
- Implemented conditional loading using Python's `os` module to ensure files were available before attempting to load.

#### 2. Data Cleaning and Preparation
- **Column Renaming**:
  - Renamed key columns for consistency between datasets.
  - The `area` column in both datasets was renamed to `area_sqft` for clarity, reflecting that the data was in square feet.
- **Data Cleaning Steps**:
  - Removed commas from `area_sqft` values and converted to numeric.
  - Processed numeric columns (`bedrooms`, `bathrooms`, `price`) by coercing invalid data to NaN and then handling missing values.
  - Addressed formatting issues such as dollar signs and non-numeric characters in the `Price` column.
  - Encoded categorical values for columns like `furnishingstatus` to ensure consistent numerical representation.

#### 3. Merging Datasets
- **Address Merging**:
  - Checked if both datasets had an `Address` column to perform a direct merge using addresses.
- **Property Feature Matching**:
  - Implemented custom property matching based on `area_sqft`, `bedrooms`, and `bathrooms` with specified tolerances for each feature.
  - Set a tolerance of `100 sqft` for `area_sqft`, `1` for `bedrooms`, and `1` for `bathrooms` to find properties that were similar but not identical.
- **Tracking Matched and Unmatched Properties**:
  - Created separate lists to store matched and unmatched properties for further analysis.
  - Counted the number of unmatched properties to determine how much of the dataset remained distinct.

#### 4. Quality Control and Error Handling
- **Handling Missing Data**:
  - Implemented missing data handling by filling numerical columns (`Price`, `area_sqft`, `bedrooms`, `bathrooms`) with their respective means.
- **Addressing `EmptyDataError`**:
  - Added error handling for empty or corrupted CSV files using Pandas' error classes.
  - Issued warnings when attempting to read an empty file and gracefully proceeded with empty DataFrames if needed.
- **Duplicate Columns**:
  - Removed duplicate columns, such as `Address Full`, to prevent redundancy in the merged dataset.


#### 5. Visualization
- **Distribution Analysis**:
  - Plotted the distribution of differences in `area_sqft` between the two datasets after merging.
  - Used histograms to visualize differences to ensure that the merged data made logical sense.
- **Pairplot Quality Check**:
  - Created pairplots for numeric columns (`Price`, `bedrooms`, `bathrooms`, `area_sqft`) to assess relationships between features.
  - Addressed duplicate label issues during visualization by dropping repeated columns.

### Outcome and Observations
- **Merged Dataset**: Successfully saved the merged dataset (`merged_housing_dataset.csv`) after removing duplicates and handling missing values.
- **Quality Control**: Completed visualization tasks to ensure the integrity of merged data, including area differences and pairwise relationships.
- **Unmatched Properties**: A significant number of properties (545) remained unmatched, indicating distinct properties in each dataset that could not be reconciled with the current tolerances.
- **Final Dataset**: The final cleaned and standardized dataset was saved as `further_cleaned_standardized_housing_dataset.csv` in the path `G:\Uni\2nd year\Intro to AI\Clustering\assignment-1-ryantigi254\Clustering\Weekly Tasks\data\`.

### Next Steps
1. **Further Cleaning**: Investigate unmatched properties and determine whether adjustments to the matching criteria could increase merge success rates.
2. **Feature Engineering**: Consider adding more features such as `Latitude` and `Longitude` for geospatial analysis to improve merging accuracy.
3. **Model Building**: Use the cleaned, merged dataset for predictive modelling or clustering tasks to derive further insights into the housing market.

### Summary
This process focused on systematically loading, cleaning, merging, and analyzing two housing datasets. Key steps included ensuring data consistency, handling missing values, dealing with empty files, merging based on property characteristics, and conducting quality control through visualizations. Each step was documented and monitored to ensure the data integrity required for subsequent analysis. The final output, `further_cleaned_standardized_housing_dataset.csv`, represents the culmination of these efforts and is ready for advanced analytical tasks.
