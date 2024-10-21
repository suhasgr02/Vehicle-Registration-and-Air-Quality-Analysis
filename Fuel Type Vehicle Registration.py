import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/suhas/Desktop/mini project/Master Thesis/Data/Main_Air_vehicle_emission_dataset.csv/Fuel type Registration of Vehicles.csv")


# Display the first few rows of the dataset to understand its structure
df.head(), df.columns

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print("Duplicate Rows:\n", duplicate_rows)

# Drop duplicates (if any)
df_cleaned = df.drop_duplicates()

# Convert 'Month' column to datetime format
df_cleaned['Month'] = pd.to_datetime(df_cleaned['Month'], format='%b-%y')

# Sort the dataset by 'Month'
df_cleaned = df_cleaned.sort_values('Month')

import matplotlib.pyplot as plt
import seaborn as sns

# Plot Electric vehicle registrations over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='ELECTRIC(BOV)', data=df_cleaned, marker='o', label='Electric Vehicles (BOV)')
plt.title('Monthly Electric Vehicle Registrations Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Registrations')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plot registrations of different fuel types over time
plt.figure(figsize=(12,6))
for column in ['CNG ONLY', 'DIESEL', 'ELECTRIC(BOV)', 'PETROL']:
    sns.lineplot(x='Month', y=column, data=df_cleaned, label=column)

plt.title('Monthly Vehicle Registrations by Fuel Type')
plt.xlabel('Month')
plt.ylabel('Number of Registrations')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Descriptive statistics for all vehicle types
stats = df_cleaned.describe()
print("Descriptive Statistics for all Fuel Types:\n", stats)

# Calculate the percentage change in electric vehicle registrations
df_cleaned['Electric_Growth_Rate'] = df_cleaned['ELECTRIC(BOV)'].pct_change() * 100

# Plot the growth rate of electric vehicles
plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='Electric_Growth_Rate', data=df_cleaned, marker='o', label='Electric Vehicle Growth Rate')
plt.title('Growth Rate of Electric Vehicle Registrations')
plt.xlabel('Month')
plt.ylabel('Growth Rate (%)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model on electric vehicle registrations
model = ARIMA(df_cleaned['ELECTRIC(BOV)'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next 12 months of electric vehicle registrations
forecast = model_fit.forecast(steps=12)
print("Forecast for Electric Vehicle Registrations:\n", forecast)

import pandas as pd
import numpy as np

# Assuming df_cleaned is already defined from previous steps

# Create a mock air quality DataFrame (Replace this with your actual air quality data)
# Make sure the length matches the vehicle registration data.
months_count = len(df_cleaned)
air_quality_data = {
    'Month': pd.date_range(start='2015-01-01', periods=months_count, freq='MS'),  # Use 'MS' for month start
    'PM2.5': np.random.randint(30, 100, size=months_count)  # Example PM2.5 data
}
air_quality_df = pd.DataFrame(air_quality_data)

# Merge the vehicle and air quality datasets on 'Month'
merged_df = pd.merge(df_cleaned, air_quality_df, on='Month')

# Ensure merged_df has data
print("Merged DataFrame Head:\n", merged_df.head())

# Check if merged_df is empty
if merged_df.empty:
    print("Merged DataFrame is empty. Please check the merging criteria.")
else:
    # Proceed with Linear Regression if merged_df has data
    from sklearn.linear_model import LinearRegression

    # Use linear regression to model the relationship between electric vehicles and air quality
    X = merged_df[['ELECTRIC(BOV)']]  # Independent variable (Electric Vehicles)
    y = merged_df['PM2.5']  # Dependent variable (Air Quality, PM2.5)

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Print model coefficients
    print(f"Intercept: {model.intercept_}, Coefficient: {model.coef_[0]}")

    # Predict air quality based on electric vehicle registrations
    predictions = model.predict(X)

    # Optionally, plot the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X['ELECTRIC(BOV)'], y=y, label='Actual PM2.5')
    sns.lineplot(x=X['ELECTRIC(BOV)'], y=predictions, color='red', label='Predicted PM2.5')
    plt.title('PM2.5 vs Electric Vehicle Registrations')
    plt.xlabel('Electric Vehicle Registrations')
    plt.ylabel('PM2.5 Levels')
    plt.legend()
    plt.show()