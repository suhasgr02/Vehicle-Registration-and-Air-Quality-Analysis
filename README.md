# Vehicle Registration and Air Quality Analysis

## Overview
This project analyzes the relationship between vehicle registrations, particularly electric vehicles, and air quality (PM2.5). Using vehicle registration data from various fuel types, the project aims to observe trends over time and to understand how the adoption of electric vehicles correlates with air quality improvement.

The analysis includes forecasting electric vehicle registrations using ARIMA and modeling the relationship between electric vehicle registrations and air quality using linear regression.

## Dataset
The main dataset used is `Fuel type Registration of Vehicles.csv`, which contains monthly vehicle registration data by fuel type. Another dataset is generated for PM2.5 levels to perform a regression analysis to model the relationship between electric vehicle adoption and air quality.

## Files
- `vehicle_registration_air_quality_analysis.py`: Main script that performs:
  - Data loading and cleaning
  - Time series plots of vehicle registrations
  - ARIMA model for forecasting electric vehicle registrations
  - Linear regression to analyze the impact of electric vehicle registrations on air quality (PM2.5)

## Requirements
To run the code, you will need the following Python libraries:
- `pandas`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scikit-learn`
- `numpy`

You can install these dependencies using:
```bash
pip install pandas matplotlib seaborn statsmodels scikit-learn numpy
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/suhasgr02/vehicle-air-quality-analysis.git
   ```
2. Place the `Fuel type Registration of Vehicles.csv` in the appropriate directory.
3. Run the Python script:
   ```bash
   python vehicle_registration_air_quality_analysis.py
   ```

## Key Steps in the Analysis
1. **Data Loading and Cleaning**: The dataset is loaded, and missing values and duplicates are checked. Duplicates are removed, and the 'Month' column is converted into a proper datetime format for time series analysis.
   
2. **Electric Vehicle Registration Trends**: A line plot visualizes the trend of electric vehicle (BOV) registrations over time. Additionally, registrations of other fuel types (CNG, Diesel, Petrol) are compared.

3. **Descriptive Statistics**: Summary statistics for all fuel types are computed, including mean, median, standard deviation, etc.

4. **Electric Vehicle Growth Rate**: The percentage change in electric vehicle registrations from month to month is calculated and plotted.

5. **Forecasting Electric Vehicle Registrations**: An ARIMA model is fitted to forecast electric vehicle registrations for the next 12 months.

6. **Air Quality Data Simulation**: A mock air quality dataset is generated with monthly PM2.5 levels to match the length of the vehicle registration data.

7. **Regression Analysis**: Linear regression is used to model the relationship between electric vehicle registrations and PM2.5 air quality data. The model's intercept and coefficient are printed, and a plot is generated to visualize actual vs predicted PM2.5 levels.

## Interpretation of Results
1. **Missing and Duplicate Data**: The analysis confirms that there are no significant missing values and duplicates are removed, ensuring data quality.
   
2. **Trends in Electric Vehicle Registrations**: The line plot shows the increasing trend of electric vehicle registrations over time, indicating growing adoption of electric vehicles.

3. **Comparative Trends of Fuel Types**: Electric vehicle registrations are plotted against other fuel types (CNG, Diesel, Petrol), providing insights into the shift toward greener energy in the automotive industry.

4. **Electric Vehicle Growth Rate**: The percentage change in electric vehicle registrations highlights fluctuations in the rate of adoption month to month. It helps identify periods of rapid growth or stagnation.

5. **Forecasting Registrations**: The ARIMA model's forecast for electric vehicle registrations gives a predictive outlook for future trends in electric vehicle adoption.

6. **Air Quality Impact**: The linear regression shows a negative relationship between electric vehicle registrations and PM2.5 levels, indicating that increasing the number of electric vehicles could be associated with improved air quality.

## Conclusion
This project provides a comprehensive analysis of vehicle registrations by fuel type over time, particularly focusing on electric vehicle adoption and its potential impact on air quality (PM2.5). By forecasting future registrations and modeling the relationship between electric vehicles and air quality, the analysis offers valuable insights into the environmental benefits of transitioning to electric vehicles.
