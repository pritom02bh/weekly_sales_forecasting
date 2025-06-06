# Weekly Sales Forecasting

Sales forecasting system using machine learning to predict sales performance with historical data, weather patterns, and temporal features. Interactive Streamlit dashboard for data analysis, model training, and forecast visualization.

## Features

- Multi-location support (4 Dutch retail locations)
- Gradient Boosting and XGBoost algorithms
- Weather data integration
- Interactive Streamlit dashboard
- Model persistence with versioning
- Daily and weekly forecasting
- Performance metrics (MAE, RMSE, R² score)
- Data validation and preprocessing

## Business Locations
- **Fenix Food Factory B.V.** (50460) - 412 records
- **Kaapse Will'ns B.V.** (47904) - 302 records  
- **Kaapse Maria B.V.** (47903) - 365 records
- **Kaapse Kaap B.V.** (47901) - 94 records

## Project Structure

```
weekly_sales_forecasting/
├── streamlit_app.py          # Main dashboard (2,835 lines)
├── sales_forecasting.py     # Core algorithms (234 lines)
├── checks.py                 # Validation (170 lines)
├── requirements.txt          # Dependencies
├── Procfile                  # Deployment
├── runtime.txt               # Python version
├── .streamlit/config.toml    # App configuration
├── data/                     # Data files
│   ├── All_Locations_Combined.csv (477 records)
│   ├── Fenix_Food_Factory_B.V._50460.csv (412 records)
│   ├── Kaapse_Will'ns_B.V._47904.csv (302 records)
│   ├── Kaapse_Maria_B.V._47903.csv (365 records)
│   ├── Kaapse_Kaap_B.V._47901.csv (94 records)
│   ├── Weather_May15_to_May27.csv
│   ├── Weather_Kaapse_Kaap_47901.csv
│   └── dataset details.pdf
└── models/                   # Model storage
    ├── model_combined_gradient_boosting_*.pkl
    ├── model_combined_xgboost_*.pkl
    └── Versioned with timestamp and feature hash
```


## Data Structure

**Sales Data**
- `Operational Date`, `Total_Sales`, `Sales_Count`, `Day_of_Week`
- `Is_Weekend`, `Is_Public_Holiday`, `Is_Closed` 
- `Tips_per_Transaction`, `Avg_Sale_per_Transaction`

**Weather Data**
- `tempmax/tempmin/temp`, `humidity`, `precip`, `precipprob`
- `cloudcover`, `solarradiation`, `uvindex`

## Machine Learning

**Models**: Gradient Boosting, XGBoost  
**Features**: Temporal, weather, sales patterns, location-specific  
**Validation**: Cross-validation with MAE, RMSE, R² metrics  
**Management**: Versioning, compression, caching, cleanup

## Dashboard

**Interface**: Company selection, date range, model configuration, feature selection  
**Visualization**: Historical trends, forecasts, performance metrics, feature importance  
**Analytics**: Time series charts, model comparison, downloadable reports

## Technical Stack

**Framework**: Streamlit  
**ML**: scikit-learn, XGBoost  
**Data**: pandas, NumPy  
**Visualization**: Plotly, Matplotlib, Seaborn  
**Storage**: joblib, compressed models with metadata

## File Details

**streamlit_app.py** - Main dashboard with caching, data management, feature engineering, model training, forecasting, and visualization

**sales_forecasting.py** - Statistical analysis, model training (Linear Regression, Random Forest), evaluation, and feature analysis

**checks.py** - Validation system for dependencies, data integrity, configuration, and deployment testing

## Deployment

Ready for: Streamlit Cloud, Heroku, Railway, Render, AWS/GCP/Azure  
Files: `Procfile`, `runtime.txt`, `requirements.txt`, `.streamlit/config.toml` 