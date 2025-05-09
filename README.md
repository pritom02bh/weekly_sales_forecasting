# Weekly Sales Forecasting

This project provides a framework for analyzing historical sales data and forecasting future weekly sales based on various factors including weather conditions, day of the week, and seasonality.

## Project Overview

The project uses March 2025 sales data along with weather information to:

1. Analyze historical sales patterns
2. Identify key factors influencing sales
3. Build predictive models (Linear Regression, Random Forest, and Gradient Boosting)
4. Generate weekly sales forecasts for April 2025

## Data

The project uses two main data sources:

- `march_data_complete.csv`: Historical sales data for March 2025, including:
  - Daily total sales
  - Transaction counts
  - Day of week information
  - Weather conditions
  - Store operating status (open/closed)
  
- `april_data.csv`: Weather forecast data for April 2025

## Features

- **Data Exploration**: Statistical analysis and visualization of historical sales data
- **Correlation Analysis**: Identifying relationships between sales and various factors
- **Feature Engineering**: Creating relevant features from date and weather data
- **Model Training**: Linear Regression, Random Forest, and Gradient Boosting models for sales prediction
- **Performance Evaluation**: Model accuracy metrics and feature importance analysis
- **Sales Forecasting**: Daily and weekly forecasts for the upcoming month
- **Visualization**: Interactive charts and graphs of historical and forecasted sales patterns

## Interactive Dashboard

The project includes a professional Streamlit dashboard that provides:

- Interactive model selection (Linear Regression, Random Forest, Gradient Boosting)
- Customizable feature selection for more accurate forecasting
- Date range selection for targeted forecasts
- Visual analysis of historical sales patterns
- Detailed model performance metrics
- Daily and weekly sales forecasts with visualizations
- Key insights and business recommendations
- Downloadable forecast data

## Usage

### Running the Dashboard Locally

To run the interactive Streamlit dashboard locally:

```
streamlit run streamlit_app.py
```

### Running the Basic Script

To run just the sales forecasting model without the dashboard:

```
python sales_forecasting.py
```

## Deployment

This application is ready for deployment to various Streamlit hosting platforms.

### Pre-Deployment Checks

Run the checks script to validate your environment before deployment:

```
python checks.py
```

### Deploying to Streamlit Cloud

1. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
2. Link your GitHub repository
3. Select the repository and the main file (`streamlit_app.py`)
4. Configure your app settings
5. Deploy

### Deploying to Heroku

1. Create a Heroku account and install the Heroku CLI
2. Log in to Heroku and create a new app
3. Push your code to Heroku:

```bash
git init
git add .
git commit -m "Initial commit"
heroku git:remote -a your-app-name
git push heroku main
```

### Deploying to Other Platforms

The application includes all necessary files for deployment to:
- [Railway](https://railway.app/)
- [Render](https://render.com/)
- [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/)
- [Google Cloud Run](https://cloud.google.com/run)

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- plotly
- joblib
- statsmodels
- scipy

## Future Improvements

- Add more sophisticated time series models (ARIMA, Prophet)
- Incorporate more historical data for improved seasonality detection
- Implement anomaly detection for unusual sales patterns
- Add confidence intervals for predictions
- Account for special events and promotions
- Incorporate inventory data for supply chain optimization 