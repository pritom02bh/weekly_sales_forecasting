import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load the data
march_data = pd.read_csv('data/forecasting_data_march.csv')
april_weather = pd.read_csv('data/april_weather.csv')

# Data exploration and preparation
def explore_data(df, name):
    print(f"\n--- {name} Data Overview ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())
    
# Explore data
explore_data(march_data, "March Sales")
explore_data(april_weather, "April Weather")

# Prepare data for modeling
# Convert date strings to datetime objects
march_data['Operational Date'] = pd.to_datetime(march_data['Operational Date'])

# Format april_weather date to match
april_weather['Operational Date'] = pd.to_datetime(april_weather['Operational Date'], format='%d-%m-%Y')

# Create daily features from the date
def extract_date_features(df):
    df['dayofweek'] = df['Operational Date'].dt.dayofweek
    df['dayofmonth'] = df['Operational Date'].dt.day
    df['week'] = df['Operational Date'].dt.isocalendar().week
    return df

march_data = extract_date_features(march_data)
april_weather = extract_date_features(april_weather)

# Data visualization
plt.figure(figsize=(12, 6))
plt.plot(march_data['Operational Date'], march_data['Total_Sales'], marker='o')
plt.title('Daily Sales for March')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('march_sales_trend.png')
plt.close()

# Weekly sales aggregation
march_data['week_of_month'] = march_data['Operational Date'].dt.day // 7 + 1
weekly_sales = march_data.groupby('week_of_month')['Total_Sales'].sum().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(weekly_sales['week_of_month'], weekly_sales['Total_Sales'])
plt.title('Weekly Sales for March')
plt.xlabel('Week of Month')
plt.ylabel('Total Sales ($)')
plt.grid(axis='y')
plt.xticks(weekly_sales['week_of_month'])
plt.tight_layout()
plt.savefig('march_weekly_sales.png')
plt.close()

# Correlation analysis
correlation_columns = ['Total_Sales', 'Sales_Count', 'Tips_per_Transaction', 
                       'Avg_Sale_per_Transaction', 'tempmax', 'tempmin', 'temp', 
                       'humidity', 'dayofweek', 'Is_Weekend']

correlation_matrix = march_data[correlation_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Sales Factors')
plt.tight_layout()
plt.savefig('sales_correlation_matrix.png')
plt.close()

# Feature selection
# Remove days when the store is closed
train_data = march_data[march_data['Is_Closed'] == 0].copy()

# Choose features for the model
features = ['dayofweek', 'Is_Weekend', 'tempmax', 'tempmin', 'temp', 
            'humidity', 'precip', 'cloudcover', 'solarradiation', 'uvindex']

X = train_data[features]
y = train_data['Total_Sales']

# Split data for model validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Model Performance ---")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred

# Evaluate regression models
lr_preds = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
rf_preds = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Sales Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Prepare April prediction data
# Assuming all April days are open except Tuesdays
april_days = pd.date_range(start='2025-04-01', end='2025-04-30')
april_prediction_data = pd.DataFrame({'Operational Date': april_days})
april_prediction_data = extract_date_features(april_prediction_data)

# Add day type features
april_prediction_data['Is_Weekend'] = (april_prediction_data['dayofweek'] >= 5).astype(int)
april_prediction_data['Is_Closed'] = (april_prediction_data['dayofweek'] == 1).astype(int)  # Tuesdays are closed

# Merge with available April weather data
april_prediction_data = april_prediction_data.merge(
    april_weather[['Operational Date', 'tempmax', 'tempmin', 'temp', 'humidity', 
                  'precip', 'cloudcover', 'solarradiation', 'uvindex']], 
    on='Operational Date', how='left'
)

# For dates without weather data, use mean values from March
for column in ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'cloudcover', 'solarradiation', 'uvindex']:
    april_prediction_data[column].fillna(march_data[column].mean(), inplace=True)

# Filter out closed days
open_days = april_prediction_data[april_prediction_data['Is_Closed'] == 0]

# Predict April sales
X_april = open_days[features]
april_predictions_lr = lr_model.predict(X_april)
april_predictions_rf = rf_model.predict(X_april)

# Add predictions to dataframe
open_days['LR_Predicted_Sales'] = april_predictions_lr
open_days['RF_Predicted_Sales'] = april_predictions_rf

# Weekly forecast
open_days['week_of_month'] = open_days['Operational Date'].dt.day // 7 + 1
weekly_forecast_lr = open_days.groupby('week_of_month')['LR_Predicted_Sales'].sum().reset_index()
weekly_forecast_rf = open_days.groupby('week_of_month')['RF_Predicted_Sales'].sum().reset_index()

# Create weekly forecast table
weekly_forecast = pd.DataFrame({
    'Week of April': weekly_forecast_lr['week_of_month'],
    'LR Weekly Forecast': weekly_forecast_lr['LR_Predicted_Sales'].round(2),
    'RF Weekly Forecast': weekly_forecast_rf['RF_Predicted_Sales'].round(2)
})

print("\n--- Weekly Sales Forecast for April 2025 ---")
print(weekly_forecast)

# Visualization of the forecast
plt.figure(figsize=(12, 6))
plt.plot(open_days['Operational Date'], open_days['RF_Predicted_Sales'], marker='o', label='Random Forest Forecast')
plt.plot(open_days['Operational Date'], open_days['LR_Predicted_Sales'], marker='x', label='Linear Regression Forecast')
plt.title('Daily Sales Forecast for April 2025')
plt.xlabel('Date')
plt.ylabel('Forecasted Sales ($)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('april_sales_forecast.png')
plt.close()

# Weekly forecast visualization
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(weekly_forecast))

plt.bar(x - bar_width/2, weekly_forecast['LR Weekly Forecast'], bar_width, label='Linear Regression')
plt.bar(x + bar_width/2, weekly_forecast['RF Weekly Forecast'], bar_width, label='Random Forest')

plt.title('Weekly Sales Forecast for April 2025')
plt.xlabel('Week of April')
plt.ylabel('Forecasted Weekly Sales ($)')
plt.xticks(x, weekly_forecast['Week of April'])
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('april_weekly_forecast.png')
plt.close()

# Summary of forecasted monthly sales
total_april_sales_lr = open_days['LR_Predicted_Sales'].sum()
total_april_sales_rf = open_days['RF_Predicted_Sales'].sum()
total_march_sales = march_data['Total_Sales'].sum()

print("\n--- Monthly Sales Summary ---")
print(f"Total March 2025 Sales: ${total_march_sales:.2f}")
print(f"Forecasted April 2025 Sales (Linear Regression): ${total_april_sales_lr:.2f}")
print(f"Forecasted April 2025 Sales (Random Forest): ${total_april_sales_rf:.2f}")
print(f"Forecasted Change (Linear Regression): {((total_april_sales_lr / total_march_sales) - 1) * 100:.2f}%")
print(f"Forecasted Change (Random Forest): {((total_april_sales_rf / total_march_sales) - 1) * 100:.2f}%") 