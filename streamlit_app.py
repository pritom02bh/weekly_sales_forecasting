import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
import os
import pathlib
import base64

# Function to resolve paths for data files
def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), "data", filename)

# Set page config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary: #4F46E5;
        --primary-light: #818CF8;
        --secondary: #10B981;
        --accent: #F59E0B;
        --background: #F9FAFB;
        --surface: #FFFFFF;
        --text: #1F2937;
        --text-light: #6B7280;
        --border: #E5E7EB;
        --error: #EF4444;
        --success: #10B981;
    }
    
    /* Modern typography */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }
    
    /* Header styling */
    .main-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--primary);
        text-align: left;
        margin-bottom: 0.25rem;
        padding-top: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: var(--text);
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        border-left: 3px solid var(--primary);
        padding-left: 0.5rem;
    }
    
    /* Modern compact title */
    .compact-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background-color: var(--surface);
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--primary);
    }
    
    /* Card styling */
    .card {
        border-radius: 8px;
        padding: 1rem;
        background-color: var(--surface);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        transition: box-shadow 0.2s ease-in-out;
    }
    
    .card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: var(--surface);
        border-left: 3px solid var(--primary);
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-light);
    }
    
    /* Insights styling */
    .insights {
        background-color: rgba(79, 70, 229, 0.05);
        border-left: 3px solid var(--primary);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    
    /* Footnote styling */
    .footnote {
        font-size: 0.7rem;
        color: var(--text-light);
        font-style: italic;
        margin-top: 0.5rem;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        border: none;
        transition: background-color 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-light);
    }
    
    /* Download button styling */
    .download-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(79, 70, 229, 0.1);
        color: var(--primary);
        border: 1px solid var(--primary);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        cursor: pointer;
        text-decoration: none;
        margin-top: 5px;
        margin-left: 10px;
        transition: background-color 0.2s ease;
    }
    
    .download-btn:hover {
        background-color: rgba(79, 70, 229, 0.2);
    }
    
    /* Chart header container */
    .chart-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Custom styling for the info boxes */
    div[data-testid="stInfo"] {
        padding: 0.5rem !important;
        background-color: rgba(79, 70, 229, 0.05) !important;
        border: 1px solid rgba(79, 70, 229, 0.2) !important;
        border-radius: 6px !important;
    }
    
    div[data-testid="stInfo"] > div {
        padding: 0 !important;
    }
    
    div[data-testid="stInfo"] p {
        font-size: 0.8rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Streamlit native element styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        background-color: #f8f9fa;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: var(--text);
        background-color: var(--background);
        border-radius: 6px;
    }
    
    /* Input field styling */
    div[data-baseweb="input"] {
        border-radius: 6px;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        border-radius: 6px;
    }
    
    /* Slider styling */
    div[data-baseweb="slider"] > div {
        background-color: var(--primary-light) !important;
    }
    
    div[data-baseweb="slider"] > div > div {
        background-color: var(--primary) !important;
    }
    
    /* Checkbox styling */
    div[data-testid="stCheckbox"] label span[aria-hidden="true"] div::before {
        border-color: var(--primary) !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        border-radius: 8px;
    }
    
    .stDataFrame th {
        background-color: rgba(79, 70, 229, 0.1) !important;
        color: var(--text) !important;
    }
    
    /* Page background */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    try:
        # Try to load with the actual dataset names
        march_data = pd.read_csv(get_data_path('march_data_complete.csv'))
        april_weather = pd.read_csv(get_data_path('april_data.csv'))
    except FileNotFoundError:
        # Fall back to alternative names if needed
        try:
            march_data = pd.read_csv(get_data_path('historical_march_data.csv'))
            april_weather = pd.read_csv(get_data_path('april_first_week.csv'))
        except FileNotFoundError:
            try:
                march_data = pd.read_csv(get_data_path('forecasting_data_march.csv'))
                april_weather = pd.read_csv(get_data_path('april_weather.csv'))
            except FileNotFoundError:
                st.error("Data files not found. Please check the data directory for correct file names.")
                st.stop()
    
    # Convert dates
    march_data['Operational Date'] = pd.to_datetime(march_data['Operational Date'])
    
    # Handle different date formats in April data
    try:
        april_weather['Operational Date'] = pd.to_datetime(april_weather['Operational Date'])
    except:
        try:
            april_weather['Operational Date'] = pd.to_datetime(april_weather['Operational Date'], format='%d-%m-%Y')
        except:
            st.warning("Date format conversion issue. Please check the date format in April dataset.")
            # Try to infer format as a last resort
            april_weather['Operational Date'] = pd.to_datetime(april_weather['Operational Date'], infer_datetime_format=True)
    
    # Don't show sidebar messages for data loading - we'll show a more compact display
    # in the main interface
    
    return march_data, april_weather

# Cache feature engineering
@st.cache_data
def engineer_features(march_data, april_weather):
    # Create a copy to avoid modifying the original
    train_data = march_data.copy()
    forecast_data = april_weather.copy()
    
    # Remove Tips data if present
    if 'Tips_per_Transaction' in train_data.columns:
        train_data.drop('Tips_per_Transaction', axis=1, inplace=True)
    
    # Extract date features
    train_data['dayofweek'] = train_data['Operational Date'].dt.dayofweek
    train_data['dayofmonth'] = train_data['Operational Date'].dt.day
    train_data['week'] = train_data['Operational Date'].dt.isocalendar().week
    train_data['week_of_month'] = train_data['Operational Date'].dt.day // 7 + 1
    
    forecast_data['dayofweek'] = forecast_data['Operational Date'].dt.dayofweek
    forecast_data['dayofmonth'] = forecast_data['Operational Date'].dt.day
    forecast_data['week'] = forecast_data['Operational Date'].dt.isocalendar().week
    forecast_data['week_of_month'] = forecast_data['Operational Date'].dt.day // 7 + 1
    forecast_data['Is_Weekend'] = (forecast_data['dayofweek'] >= 5).astype(int)
    forecast_data['Is_Closed'] = (forecast_data['dayofweek'] == 1).astype(int)  # Tuesdays are closed
    
    # Ensure consistent feature names between train and forecast data
    # Some datasets might have 'temp' in one and not the other
    if 'temp' in train_data.columns and 'temp' not in forecast_data.columns:
        forecast_data['temp'] = forecast_data['tempmin'] + (forecast_data['tempmax'] - forecast_data['tempmin']) / 2
    
    if 'temp' in forecast_data.columns and 'temp' not in train_data.columns:
        train_data['temp'] = train_data['tempmin'] + (train_data['tempmax'] - train_data['tempmin']) / 2
    
    return train_data, forecast_data

# Train models with enhanced evaluation
@st.cache_resource
def train_models(train_data, features, target, model_params=None, cv_folds=5, test_size=0.2):
    """Train forecasting models with proper statistical evaluation."""
    # Remove closed days
    train_data = train_data[train_data['Is_Closed'] == 0].copy()
    
    X = train_data[features]
    y = train_data[target]
    
    # Handle potential missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Default parameters if none provided
    if model_params is None:
        model_params = {
            'XGBoost': {'n_estimators': 100, 'random_state': 42},
            'Gradient Boosting': {'n_estimators': 100, 'random_state': 42}
        }
    
    # Initialize models
    models = {
        'XGBoost': XGBRegressor(**model_params.get('XGBoost', {'n_estimators': 100, 'random_state': 42})),
        'Gradient Boosting': GradientBoostingRegressor(**model_params.get('Gradient Boosting', {'n_estimators': 100, 'random_state': 42}))
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100  # Mean Absolute Percentage Error
        
        # Comprehensive cross-validation
        cv_scores_mae = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
        cv_scores_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        # Calculate confidence intervals using bootstrap
        bootstrap_predictions = []
        indices = np.arange(len(X_test))
        
        for _ in range(100):  # 100 bootstrap samples
            bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
            X_bootstrap = X_test.iloc[bootstrap_indices]
            bootstrap_predictions.append(model.predict(X_bootstrap))
            
        bootstrap_predictions = np.array(bootstrap_predictions)
        lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        # Store all results
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'cv_mae': -np.mean(cv_scores_mae),
            'cv_r2': np.mean(cv_scores_r2),
            'cv_mae_std': np.std(cv_scores_mae),
            'cv_r2_std': np.std(cv_scores_r2),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'lower_ci_factor': np.mean(y_pred - lower_ci),
            'upper_ci_factor': np.mean(upper_ci - y_pred)
        }
        
        # Check for overfitting
        results[name]['overfitting'] = results[name]['train_score'] - results[name]['test_score']
    
    return results

# Generate forecast with confidence intervals and historical data
def generate_forecast(model_results, historical_data, forecast_data, features, selected_model):
    """Generate forecast with confidence intervals and include historical data for comparison."""
    # Create copies of the data to avoid modifying the originals
    historical_data_copy = historical_data.copy()
    forecast_data_copy = forecast_data.copy()
    
    # Keep all days but mark closed days for special handling
    historical_data_copy['Is_Open'] = (historical_data_copy['Is_Closed'] == 0).astype(int)
    forecast_data_copy['Is_Open'] = (forecast_data_copy['Is_Closed'] == 0).astype(int)
    
    # For open days, prepare features for prediction
    historical_open = historical_data_copy[historical_data_copy['Is_Open'] == 1]
    forecast_open = forecast_data_copy[forecast_data_copy['Is_Open'] == 1]
    
    # Handle potential missing values
    X_historical = historical_open[features].fillna(historical_open[features].mean())
    X_forecast = forecast_open[features].fillna(forecast_open[features].mean())
    
    # Get the selected model
    model = model_results[selected_model]['model']
    
    # Make forecasts for open days
    historical_predicted = model.predict(X_historical)
    forecasted_sales = model.predict(X_forecast)
    
    # Add predictions to the open days
    historical_open_with_pred = historical_open.copy()
    historical_open_with_pred['Predicted_Sales'] = historical_predicted
    historical_open_with_pred['Actual_Sales'] = historical_open_with_pred['Total_Sales']
    
    forecast_open_with_pred = forecast_open.copy()
    forecast_open_with_pred['Forecasted_Sales'] = forecasted_sales
    
    # Add confidence intervals based on model metrics
    mae = model_results[selected_model]['mae']
    lower_ci_factor = model_results[selected_model].get('lower_ci_factor', mae * 1.96)
    upper_ci_factor = model_results[selected_model].get('upper_ci_factor', mae * 1.96)
    
    # Historical confidence intervals (for validation)
    historical_open_with_pred['Lower_Bound'] = historical_open_with_pred['Predicted_Sales'] - lower_ci_factor
    historical_open_with_pred['Upper_Bound'] = historical_open_with_pred['Predicted_Sales'] + upper_ci_factor
    historical_open_with_pred['Absolute_Error'] = np.abs(historical_open_with_pred['Actual_Sales'] - historical_open_with_pred['Predicted_Sales'])
    historical_open_with_pred['Within_CI'] = (
        (historical_open_with_pred['Actual_Sales'] >= historical_open_with_pred['Lower_Bound']) & 
        (historical_open_with_pred['Actual_Sales'] <= historical_open_with_pred['Upper_Bound'])
    )
    
    # Forecast confidence intervals
    forecast_open_with_pred['Lower_Bound'] = forecast_open_with_pred['Forecasted_Sales'] - lower_ci_factor
    forecast_open_with_pred['Upper_Bound'] = forecast_open_with_pred['Forecasted_Sales'] + upper_ci_factor
    
    # Ensure no negative sales predictions or bounds
    historical_open_with_pred['Lower_Bound'] = historical_open_with_pred['Lower_Bound'].clip(lower=0)
    forecast_open_with_pred['Lower_Bound'] = forecast_open_with_pred['Lower_Bound'].clip(lower=0)
    
    # Now merge the open days back with the closed days
    
    # For closed days (historical), set predicted values to 0
    historical_closed = historical_data_copy[historical_data_copy['Is_Open'] == 0].copy()
    if not historical_closed.empty:
        historical_closed['Predicted_Sales'] = 0
        historical_closed['Actual_Sales'] = 0
        historical_closed['Lower_Bound'] = 0
        historical_closed['Upper_Bound'] = 0
        historical_closed['Absolute_Error'] = 0
        historical_closed['Within_CI'] = True
    
    # For closed days (forecast), set predicted values to 0
    forecast_closed = forecast_data_copy[forecast_data_copy['Is_Open'] == 0].copy()
    if not forecast_closed.empty:
        forecast_closed['Forecasted_Sales'] = 0
        forecast_closed['Lower_Bound'] = 0
        forecast_closed['Upper_Bound'] = 0
    
    # IMPORTANT: Ensure continuity by making the last day of March connect to the first day of April
    # We'll create a consistent value for last/first days to ensure visual continuity
    if not historical_open.empty and not forecast_open.empty:
        # Get the last historical date
        last_historical_date = historical_data_copy['Operational Date'].max()
        next_forecast_date = forecast_data_copy['Operational Date'].min()
        
        # Find the indices for these dates in their respective dataframes
        if last_historical_date in historical_open_with_pred['Operational Date'].values:
            last_hist_idx = historical_open_with_pred[historical_open_with_pred['Operational Date'] == last_historical_date].index[0]
            historical_open_with_pred.loc[last_hist_idx, 'Predicted_Sales'] = historical_predicted[-1]
        
        if next_forecast_date in forecast_open_with_pred['Operational Date'].values:
            first_forecast_idx = forecast_open_with_pred[forecast_open_with_pred['Operational Date'] == next_forecast_date].index[0]
            # Use a value that creates visual continuity
            forecast_open_with_pred.loc[first_forecast_idx, 'Forecasted_Sales'] = forecasted_sales[0]
        
    # Combine open and closed days
    historical_all = pd.concat([historical_open_with_pred, historical_closed], ignore_index=False)
    forecast_all = pd.concat([forecast_open_with_pred, forecast_closed], ignore_index=False)
    
    # Sort by date
    historical_all = historical_all.sort_values('Operational Date')
    forecast_all = forecast_all.sort_values('Operational Date')
    
    # Calculate weekly aggregates for historical data
    historical_weekly = historical_all.groupby('week_of_month').agg({
        'Actual_Sales': 'sum',
        'Predicted_Sales': 'sum',
        'Lower_Bound': 'sum',
        'Upper_Bound': 'sum'
    }).reset_index()
    historical_weekly.rename(columns={'week_of_month': 'Week of Month'}, inplace=True)
    
    # Calculate weekly aggregates for forecast data
    forecast_weekly = forecast_all.groupby('week_of_month').agg({
        'Forecasted_Sales': 'sum',
        'Lower_Bound': 'sum',
        'Upper_Bound': 'sum'
    }).reset_index()
    forecast_weekly.rename(columns={'week_of_month': 'Week of Month'}, inplace=True)
    
    return historical_all, forecast_all, historical_weekly, forecast_weekly

# Feature importance
def get_feature_importance(model_results, features, selected_model):
    model = model_results[selected_model]['model']
    
    if selected_model == 'XGBoost':
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    return importance

# Set a consistent color palette for all visualizations
def get_color_palette():
    return {
        'primary': '#4F46E5',       # Modern indigo primary
        'secondary': '#818CF8',     # Lighter indigo
        'accent': '#F59E0B',        # Amber accent
        'positive': '#10B981',      # Emerald green
        'negative': '#EF4444',      # Red for negative
        'neutral': '#6B7280',       # Gray neutral
        'background': '#F9FAFB',    # Very light background
        'grid': '#E5E7EB',          # Light grid lines
        'text': '#1F2937',          # Dark text
    }

# Enhanced visualization function for line charts
def create_line_chart(data, x, y, title, xlabel, ylabel, markers=True, color=None):
    colors = get_color_palette()
    color = color or colors['primary']
    
    fig = px.line(data, x=x, y=y, 
                 title=title,
                 labels={y: ylabel, x: xlabel},
                 markers=markers)
    
    fig.update_traces(line=dict(color=color, width=3), 
                     marker=dict(size=8, color=color))
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Enhanced visualization function for bar charts
def create_bar_chart(data, x, y, title, xlabel, ylabel, color=None):
    colors = get_color_palette()
    color = color or colors['primary']
    
    fig = px.bar(data, x=x, y=y, 
               title=title,
               labels={y: ylabel, x: xlabel},
               color_discrete_sequence=[color])
    
    fig.update_traces(
        marker_line_width=0,
        opacity=0.9,
        hovertemplate='%{x}: <b>$%{y:.2f}</b><extra></extra>'
    )
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Modified version of create_scatter_chart to avoid statsmodels dependency
def create_scatter_chart(data, x, y, title, xlabel, ylabel, add_trendline=False, color=None):
    """Create a scatter plot without trendline to avoid statsmodels dependency."""
    colors = get_color_palette()
    color = color or colors['primary']
    
    fig = px.scatter(data, x=x, y=y, 
                    title=title,
                    labels={y: ylabel, x: xlabel},
                    color_discrete_sequence=[color],
                    trendline=None)  # Remove trendline to avoid statsmodels dependency
    
    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Enhanced line chart with confidence interval
def create_line_chart_with_ci(data, x, y, lower_bound, upper_bound, title, xlabel, ylabel, markers=True, color=None):
    colors = get_color_palette()
    color = color or colors['primary']
    
    fig = go.Figure()
    
    # Check if we have day names available
    has_day_names = 'Date_With_Day' in data.columns
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=data[x].tolist() + data[x].tolist()[::-1],
        y=data[upper_bound].tolist() + data[lower_bound].tolist()[::-1],
        fill='toself',
        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    # Create customdata for hover labels if day names are available
    customdata = data['Date_With_Day'].tolist() if has_day_names else None
    hovertemplate = '%{customdata}<br>$%{y:.2f}<extra>Forecast</extra>' if has_day_names else None
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode='lines+markers' if markers else 'lines',
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        name="Forecast",
        customdata=customdata,
        hovertemplate=hovertemplate
    ))
    
    # Create custom tick labels with day names if available
    if has_day_names:
        # Use every date to avoid gaps, but adjust the angle and size as needed
        fig.update_xaxes(
            tickvals=data[x].tolist(),
            ticktext=data['Date_With_Day'].tolist(),
            tickangle=45,
            tickfont=dict(size=10)
        )
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=11),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif"),
        hovermode="x unified"
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Enhanced bar chart with error bars
def create_bar_chart_with_error(data, x, y, lower_bound, upper_bound, title, xlabel, ylabel, color=None):
    colors = get_color_palette()
    color = color or colors['primary']
    
    # Calculate error values
    error_y = [data[y] - data[lower_bound], data[upper_bound] - data[y]]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=data[x],
        y=data[y],
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y[1],
            arrayminus=error_y[0],
            color=colors['accent'],
            thickness=1.5,
            width=6
        ),
        marker_color=color,
        opacity=0.9,
        hovertemplate='%{x}: <b>$%{y:.2f}</b><br>Range: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<extra></extra>',
        customdata=np.column_stack((data[lower_bound], data[upper_bound]))
    ))
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Function to save a trained model
def save_model(model_results, model_name, features):
    """Save a trained model to disk."""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info = {
        'model': model_results[model_name]['model'],
        'metrics': {k: v for k, v in model_results[model_name].items() if k != 'model'},
        'features': features,
        'timestamp': timestamp,
        'model_type': model_name
    }
    
    filename = f"models/model_{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
    joblib.dump(model_info, filename)
    return filename

# Function to load a trained model
def load_model(filename):
    """Load a trained model from disk."""
    try:
        model_info = joblib.load(filename)
        return model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to list saved models
def list_saved_models():
    """List all saved models in the models directory."""
    if not os.path.exists('models'):
        return []
    
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    models_info = []
    
    for model_file in model_files:
        try:
            model_path = os.path.join('models', model_file)
            model_info = joblib.load(model_path)
            
            # Extract key information
            models_info.append({
                'filename': model_file,
                'model_type': model_info['model_type'],
                'timestamp': model_info['timestamp'],
                'features': len(model_info['features']),
                'r2_score': model_info['metrics']['r2'],
                'path': model_path
            })
        except Exception as e:
            st.warning(f"Could not load model info for {model_file}: {e}")
    
    return models_info

# Create a combined historical and forecast visualization
def create_historical_forecast_chart(historical_data, forecast_data, 
                                    x_col='Operational Date', 
                                    historical_y_cols=('Actual_Sales', 'Predicted_Sales'),
                                    forecast_y_col='Forecasted_Sales',
                                    historical_lower='Lower_Bound',
                                    historical_upper='Upper_Bound',
                                    forecast_lower='Lower_Bound',
                                    forecast_upper='Upper_Bound',
                                    title="Sales Trend: Historical Data and Forecast", 
                                    xlabel="Date", 
                                    ylabel="Sales ($)"):
    """Create a combined visualization of historical data and forecast."""
    colors = get_color_palette()
    fig = go.Figure()
    
    # Check if we have day names available
    has_day_names = 'Date_With_Day' in historical_data.columns and 'Date_With_Day' in forecast_data.columns
    
    # Create continuous visualization strategy:
    # 1. First plot all data (including closed days) with a lightweight connecting line
    # 2. Then overlay open day data with proper styling
    # 3. Finally mark closed days with special symbols
    
    # Make a copy of all data for manipulation
    all_historical = historical_data.copy()
    all_forecast = forecast_data.copy()
    
    # Ensure closed days have zero sales values
    if 'Is_Open' in all_historical.columns:
        all_historical.loc[all_historical['Is_Open'] == 0, historical_y_cols[0]] = 0
        all_historical.loc[all_historical['Is_Open'] == 0, historical_y_cols[1]] = 0
        
    if 'Is_Open' in all_forecast.columns:
        all_forecast.loc[all_forecast['Is_Open'] == 0, forecast_y_col] = 0
    
    # Sort by date to ensure continuous line
    all_historical = all_historical.sort_values(x_col)
    all_forecast = all_forecast.sort_values(x_col)
    
    # STEP 1: Add a lightweight connecting line for ALL days (continuous line)
    # First, combine the historical and forecast data to create one continuous line
    if not all_historical.empty and not all_forecast.empty:
        # Get the last historical date and first forecast date
        last_historical_date = all_historical[x_col].max()
        last_historical_value = all_historical[all_historical[x_col] == last_historical_date][historical_y_cols[1]].values[0]
        
        first_forecast_date = all_forecast[x_col].min()
        first_forecast_value = all_forecast[all_forecast[x_col] == first_forecast_date][forecast_y_col].values[0]
        
        # Create a continuous line from history to forecast
        # First plot all historical data
        fig.add_trace(go.Scatter(
            x=all_historical[x_col],
            y=all_historical[historical_y_cols[0]],
            mode='lines',
            name='Actual Sales (All Days)',
            line=dict(color=colors['primary'], width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Plot historical predictions
        fig.add_trace(go.Scatter(
            x=all_historical[x_col],
            y=all_historical[historical_y_cols[1]],
            mode='lines',
            name='Model Prediction (All Days)',
            line=dict(color=colors['secondary'], width=1.5, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Create a bridge between historical predictions and forecast
        # This ensures visual continuity between the two periods
        bridge_x = [last_historical_date, first_forecast_date]
        bridge_y = [last_historical_value, first_forecast_value]
        
        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode='lines',
            name='Connection Bridge',
            line=dict(color=colors['accent'], width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Then plot all forecast data
        fig.add_trace(go.Scatter(
            x=all_forecast[x_col],
            y=all_forecast[forecast_y_col],
            mode='lines',
            name='Forecast (All Days)',
            line=dict(color=colors['accent'], width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    else:
        # If we only have one set of data, plot it normally
        if not all_historical.empty:
            fig.add_trace(go.Scatter(
                x=all_historical[x_col],
                y=all_historical[historical_y_cols[0]],
                mode='lines',
                name='Actual Sales (All Days)',
                line=dict(color=colors['primary'], width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=all_historical[x_col],
                y=all_historical[historical_y_cols[1]],
                mode='lines',
                name='Model Prediction (All Days)',
                line=dict(color=colors['secondary'], width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        if not all_forecast.empty:
            fig.add_trace(go.Scatter(
                x=all_forecast[x_col],
                y=all_forecast[forecast_y_col],
                mode='lines',
                name='Forecast (All Days)',
                line=dict(color=colors['accent'], width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
            
    # STEP 2: Now overlay the open days with proper styling
    if not historical_data.empty:
        # Separate open and closed days for historical data
        historical_open = historical_data[historical_data['Is_Open'] == 1].copy() if 'Is_Open' in historical_data.columns else historical_data.copy()
        historical_closed = historical_data[historical_data['Is_Open'] == 0].copy() if 'Is_Open' in historical_data.columns else pd.DataFrame()
        
        # Add historical actual sales for open days
        customdata = historical_open['Date_With_Day'].tolist() if has_day_names else None
        hovertemplate = '%{customdata}<br>$%{y:.2f}<extra>Actual Sales</extra>' if has_day_names else None
        
        fig.add_trace(go.Scatter(
            x=historical_open[x_col],
            y=historical_open[historical_y_cols[0]],
            mode='markers',  # Only markers, the connecting line was added above
            name='Actual Sales (March)',
            marker=dict(size=8, color=colors['primary']),
            customdata=customdata,
            hovertemplate=hovertemplate
        ))
        
        # Add historical model prediction for open days
        customdata = historical_open['Date_With_Day'].tolist() if has_day_names else None
        hovertemplate = '%{customdata}<br>$%{y:.2f}<extra>Predicted</extra>' if has_day_names else None
        
        fig.add_trace(go.Scatter(
            x=historical_open[x_col],
            y=historical_open[historical_y_cols[1]],
            mode='markers',  # Only markers, the connecting line was added above
            name='Model Prediction (March)',
            marker=dict(size=6, color=colors['secondary']),
            customdata=customdata,
            hovertemplate=hovertemplate
        ))
        
        # Add historical confidence interval if provided
        if historical_lower and historical_upper and historical_lower in historical_open.columns and historical_upper in historical_open.columns:
            fig.add_trace(go.Scatter(
                x=historical_open[x_col].tolist() + historical_open[x_col].tolist()[::-1],
                y=historical_open[historical_upper].tolist() + historical_open[historical_lower].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(colors["secondary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo="skip",
                name='95% CI (March)'
            ))
    
    # Ensure we have forecast data to plot
    if not forecast_data.empty:
        # Separate open and closed days for forecast data
        forecast_open = forecast_data[forecast_data['Is_Open'] == 1].copy() if 'Is_Open' in forecast_data.columns else forecast_data.copy()
        forecast_closed = forecast_data[forecast_data['Is_Open'] == 0].copy() if 'Is_Open' in forecast_data.columns else pd.DataFrame()
        
        # Add forecast sales for open days
        customdata = forecast_open['Date_With_Day'].tolist() if has_day_names else None
        hovertemplate = '%{customdata}<br>$%{y:.2f}<extra>Forecast</extra>' if has_day_names else None
        
        fig.add_trace(go.Scatter(
            x=forecast_open[x_col],
            y=forecast_open[forecast_y_col],
            mode='markers',  # Only markers, the connecting line was added above
            name='Forecast (April)',
            marker=dict(size=10, color=colors['accent']),
            customdata=customdata,
            hovertemplate=hovertemplate
        ))
        
        # Add forecast confidence interval if provided
        if forecast_lower and forecast_upper and forecast_lower in forecast_open.columns and forecast_upper in forecast_open.columns:
            fig.add_trace(go.Scatter(
                x=forecast_open[x_col].tolist() + forecast_open[x_col].tolist()[::-1],
                y=forecast_open[forecast_upper].tolist() + forecast_open[forecast_lower].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(colors["accent"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo="skip",
                name='95% CI (April)'
            ))
    
    # STEP 3: Mark closed days with special symbols
    if not historical_data.empty and 'Is_Open' in historical_data.columns:
        historical_closed = historical_data[historical_data['Is_Open'] == 0].copy()
        if not historical_closed.empty:
            customdata_closed = historical_closed['Date_With_Day'].tolist() if has_day_names else None
            hovertemplate_closed = '%{customdata}<br>Closed Day<extra></extra>' if has_day_names else None
            
            fig.add_trace(go.Scatter(
                x=historical_closed[x_col],
                y=[0] * len(historical_closed),
                mode='markers',
                name='Closed Days (March)',
                marker=dict(
                    symbol='x',
                    size=10,
                    color=colors['negative'],
                    line=dict(width=2, color=colors['negative'])
                ),
                customdata=customdata_closed,
                hovertemplate=hovertemplate_closed
            ))
    
    if not forecast_data.empty and 'Is_Open' in forecast_data.columns:
        forecast_closed = forecast_data[forecast_data['Is_Open'] == 0].copy()
        if not forecast_closed.empty:
            customdata_closed = forecast_closed['Date_With_Day'].tolist() if has_day_names else None
            hovertemplate_closed = '%{customdata}<br>Closed Day<extra></extra>' if has_day_names else None
            
            fig.add_trace(go.Scatter(
                x=forecast_closed[x_col],
                y=[0] * len(forecast_closed),
                mode='markers',
                name='Closed Days (April)',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='rgba(255, 0, 0, 0.7)',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.7)')
                ),
                customdata=customdata_closed,
                hovertemplate=hovertemplate_closed
            ))
    
    # Only add vertical separator line if we have both historical and forecast data
    if not historical_data.empty and not forecast_data.empty:
        # Add vertical line to separate historical from forecast
        last_historical_date = historical_data[x_col].max()
        first_forecast_date = forecast_data[x_col].min()
        
        middle_date = last_historical_date + (first_forecast_date - last_historical_date) / 2
        
        fig.add_shape(
            type="line",
            x0=middle_date,
            y0=0,
            x1=middle_date,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
        )
        
        # Add annotation to mark the separation
        fig.add_annotation(
            x=middle_date,
            y=1,
            yref="paper",
            text="Historical | Forecast",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="white",
            opacity=0.8,
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    # Create custom tick labels with day names if available
    if has_day_names and (not historical_data.empty or not forecast_data.empty):
        # Combine historical and forecast data dates and tick labels
        all_dates = []
        all_labels = []
        
        if not historical_data.empty:
            all_dates.extend(list(historical_data[x_col]))
            all_labels.extend(list(historical_data['Date_With_Day']))
            
        if not forecast_data.empty:
            all_dates.extend(list(forecast_data[x_col]))
            all_labels.extend(list(forecast_data['Date_With_Day']))
        
        # We might not want to show every single date label to avoid overcrowding
        # Instead, let's show every 3-4 days to keep it readable
        step_size = max(1, len(all_dates) // 12)  # Show about 12 labels total
        date_indices = list(range(0, len(all_dates), step_size))
        tick_vals = [all_dates[i] for i in date_indices if i < len(all_dates)]
        tick_text = [all_labels[i] for i in date_indices if i < len(all_labels)]
        
        # Update x-axis with the custom tick labels
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45
        )
    
    # Add annotations for closed days
    if not historical_data.empty and 'Is_Open' in historical_data.columns:
        historical_closed = historical_data[historical_data['Is_Open'] == 0]
        for _, row in historical_closed.iterrows():
            date_str = row['Date_With_Day'] if 'Date_With_Day' in row else row[x_col].strftime('%Y-%m-%d')
            fig.add_annotation(
                x=row[x_col],
                y=0,
                text="CLOSED",
                showarrow=False,
                font=dict(color="red", size=8),
                yshift=-15
            )
    
    if not forecast_data.empty and 'Is_Open' in forecast_data.columns:
        forecast_closed = forecast_data[forecast_data['Is_Open'] == 0]
        for _, row in forecast_closed.iterrows():
            date_str = row['Date_With_Day'] if 'Date_With_Day' in row else row[x_col].strftime('%Y-%m-%d')
            fig.add_annotation(
                x=row[x_col],
                y=0,
                text="CLOSED",
                showarrow=False,
                font=dict(color="red", size=8),
                yshift=-15
            )
    
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=500,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=11),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=colors['grid'],
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12
        ),
        font=dict(family="Arial, sans-serif"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add a subtle background color
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=0),
        fillcolor=colors['background'],
        layer="below",
        opacity=0.1
    )
    
    return fig

# Main dashboard with full view of historical data and forecast
def main():
    # Create a compact header with inline data info
    header_cols = st.columns([3, 1, 1])
    with header_cols[0]:
        st.markdown("""
        <div class='compact-title'>
            <div style="display: flex; align-items: center;">
                <div style="background-color: rgba(79, 70, 229, 0.1); border-radius: 8px; width: 36px; height: 36px; display: flex; justify-content: center; align-items: center; margin-right: 12px;">
                    <span style="font-size: 20px;">📊</span>
                </div>
                <div class='main-header'>Sales Forecasting Dashboard</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data first
    march_data, april_weather = load_data()
    
    # Show compact data loading info in the same row as the header
    with header_cols[1]:
        st.markdown("""
        <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 6px; padding: 8px 12px; margin-top: 8px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 14px; font-weight: 500; color: #10B981;">📅 March</span>
                <span style="margin-left: auto; font-weight: 600; color: #1F2937;">{}</span>
            </div>
        </div>
        """.format(march_data.shape[0]), unsafe_allow_html=True)
    
    with header_cols[2]:
        st.markdown("""
        <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 6px; padding: 8px 12px; margin-top: 8px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 14px; font-weight: 500; color: #F59E0B;">📅 April</span>
                <span style="margin-left: auto; font-weight: 600; color: #1F2937;">{}</span>
            </div>
        </div>
        """.format(april_weather.shape[0]), unsafe_allow_html=True)
        
    # Create a layout with narrower configuration area and wider forecast area
    col_config, col_forecast = st.columns([1, 4])
    
    with col_config:
        st.markdown("<div class='sub-header' style='font-size: 1.3rem;'>Configuration</div>", unsafe_allow_html=True)
        
        with st.expander("⚙️ Forecast Settings", expanded=True):
            # Engineer features
            train_data, forecast_data = engineer_features(march_data, april_weather)
            
            # Get min and max dates from the April data
            april_min_date = forecast_data['Operational Date'].min()
            april_max_date = forecast_data['Operational Date'].max()
            
            # Ultra-compact layout with two columns for dates
            st.markdown("<div style='margin-bottom: 12px; font-weight: 500; color: var(--text);'>Forecast Period</div>", unsafe_allow_html=True)
            col_dates = st.columns(2)
            with col_dates[0]:
                st.markdown("<div style='font-size: 0.85rem; color: var(--text-light);'>Start Date</div>", unsafe_allow_html=True)
                forecast_start = st.date_input(
                    "Start Date",
                    value=april_min_date,
                    min_value=april_min_date,
                    max_value=april_max_date,
                    key="start_date",
                    label_visibility="collapsed"
                )
            
            with col_dates[1]:
                st.markdown("<div style='font-size: 0.85rem; color: var(--text-light);'>End Date</div>", unsafe_allow_html=True)
                forecast_end = st.date_input(
                    "End Date",
                    value=april_max_date,
                    min_value=forecast_start,
                    max_value=april_max_date,
                    key="end_date",
                    label_visibility="collapsed"
                )
            
            # Convert to datetime
            forecast_start = pd.to_datetime(forecast_start)
            forecast_end = pd.to_datetime(forecast_end)
            
            st.markdown("<div style='margin: 12px 0; border-top: 1px solid var(--border);'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 12px; font-weight: 500; color: var(--text);'>Display Options</div>", unsafe_allow_html=True)
            
            # Ultra-compact display options with checkboxes in the same line
            col_opts = st.columns(2)
            with col_opts[0]:
                include_historical = st.checkbox("Show March Data", value=True)
            with col_opts[1]:
                confidence_interval = st.checkbox("Show CI", value=True)
            
            st.markdown("<div style='margin: 12px 0; border-top: 1px solid var(--border);'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 12px; font-weight: 500; color: var(--text);'>Model Selection</div>", unsafe_allow_html=True)
            
            # Model selection more compact
            model_option = st.selectbox(
                "Model",
                ["XGBoost", "Gradient Boosting"],
                index=1,
                label_visibility="collapsed"
            )
            
            # Use all available features instead of allowing selection
            available_features = ['dayofweek', 'dayofmonth', 'week', 'Is_Weekend', 
                                'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                                'precipprob', 'cloudcover', 'solarradiation', 'uvindex']
            
            # Remove Tips data from available features
            if 'Tips_per_Transaction' in available_features:
                available_features.remove('Tips_per_Transaction')
            
            # Only include features that exist in the dataset
            selected_features = [feature for feature in available_features if feature in train_data.columns or feature in ['dayofweek', 'dayofmonth', 'week', 'Is_Weekend']]
            
            # Ensure all necessary features are included
            if 'temp' in train_data.columns and 'temp' not in selected_features:
                selected_features.append('temp')
                
            # Show information about feature count in a smaller way
            st.caption(f"Using all {len(selected_features)} available features")
            
            # Show save model option only
            save_model_checkbox = st.checkbox("Save model after training", value=False)
        
        # Advanced options hidden by default and made much more compact
        with st.expander("🔍 Advanced", expanded=False):
            model_params = {}
            
            if model_option == "XGBoost":
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.number_input("Trees", min_value=50, max_value=500, value=100, step=10)
                with col2:
                    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01, format="%.2f")
                
                col3, col4 = st.columns(2)
                with col3:
                    max_depth = st.number_input("Max Depth", min_value=3, max_value=15, value=6, step=1)
                with col4:
                    subsample = st.number_input("Subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.1, format="%.1f")
                
                model_params = {
                    "n_estimators": n_estimators, 
                    "learning_rate": learning_rate, 
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "random_state": 42
                }
            
            elif model_option == "Gradient Boosting":
                col1, col2 = st.columns(2)
                with col1:
                    n_estimators = st.number_input("Estimators", min_value=50, max_value=500, value=100, step=10)
                with col2:
                    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01, format="%.2f")
                model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "random_state": 42}
            
            # Cross-validation fold in the same row as another parameter
            cv_folds = st.number_input("CV Folds", min_value=3, max_value=10, value=5, step=1)
        
        # Generate Forecast button - make it stand out more
        st.markdown("<div style='margin: 20px 0 10px 0;'></div>", unsafe_allow_html=True)
        generate_btn = st.button("▶️ Generate Forecast", type="primary", use_container_width=True, key="generate_forecast_btn")
        
        # Add model management in a collapsed section - made more compact
        with st.expander("💾 Models", expanded=False):
            saved_models = list_saved_models()
            
            if not saved_models:
                st.info("No saved models found")
            else:
                # Ultra compact model display - just show the essential info
                display_df = pd.DataFrame([
                    {'Model': m['model_type'], 
                     'Date': pd.to_datetime(m['timestamp'], format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d'), 
                     'R²': round(m['r2_score'], 3)} 
                    for m in saved_models
                ])
                
                # Display a very compact models table
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=100, key="saved_models_table")
                
                # Compact model selection and actions in a single row
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    selected_model_index = st.selectbox(
                        "Model",
                        range(len(saved_models)),
                        format_func=lambda i: f"{saved_models[i]['model_type']} ({pd.to_datetime(saved_models[i]['timestamp'], format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d')})",
                        label_visibility="collapsed"
                    )
                    selected_model_path = saved_models[selected_model_index]['path']
                    
                with cols[1]:
                    load_btn = st.button("Load", use_container_width=True, key="load_model_btn")
                    
                with cols[2]:
                    delete_btn = st.button("Delete", use_container_width=True, key="delete_model_btn")
                
                # Handle load model
                if load_btn:
                    loaded_model_info = load_model(selected_model_path)
                    if loaded_model_info:
                        st.session_state.loaded_model_info = loaded_model_info
                        st.success(f"Loaded {loaded_model_info['model_type']} model")
                
                # Handle delete model
                if delete_btn:
                    try:
                        os.remove(selected_model_path)
                        st.success(f"Deleted model")
                        # Force refresh
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting model: {e}")
    
    with col_forecast:
        # Only show forecast section when button is clicked
        if generate_btn or 'model_results' in st.session_state:
            with st.spinner("Training model and generating forecast..."):
                # Check if we should use a loaded model
                use_loaded_model = False
                if 'loaded_model_info' in st.session_state:
                    use_loaded_model = st.checkbox("Use loaded model", value=True, help="Use previously loaded model instead of training a new one")
                
                # Either use loaded model or train a new one
                if use_loaded_model and 'loaded_model_info' in st.session_state:
                    # Use the loaded model
                    loaded_model_info = st.session_state.loaded_model_info
                    
                    # Check if features match
                    model_features = loaded_model_info['features']
                    if not all(feature in selected_features for feature in model_features):
                        st.warning(f"⚠️ Model trained with different features")
                    
                    # Create session_state.model_results structure
                    model_type = loaded_model_info['model_type']
                    st.session_state.model_results = {
                        model_type: {
                            'model': loaded_model_info['model'],
                            **loaded_model_info['metrics']
                        }
                    }
                    
                    # Set the model option to match the loaded model
                    model_option = model_type
                else:
                    # Train a new model
                    if 'model_results' not in st.session_state or generate_btn:
                        # Create dictionary for model parameters
                        current_model_params = {
                            model_option: model_params if 'model_params' in locals() else {}
                        }
                        
                        # Train models
                        st.session_state.model_results = train_models(
                            train_data, 
                            selected_features, 
                            'Total_Sales',
                            model_params=current_model_params,
                            cv_folds=cv_folds if 'cv_folds' in locals() else 5
                        )
                        
                        # Save the model if requested
                        if save_model_checkbox:
                            saved_model_path = save_model(st.session_state.model_results, model_option, selected_features)
                            st.success(f"Model saved")
                
                # Generate forecast for selected date range
                forecast_data_filtered = forecast_data[
                    (forecast_data['Operational Date'] >= forecast_start) & 
                    (forecast_data['Operational Date'] <= forecast_end)
                ]
                
                # Include all March data for historical comparison
                historical_all, forecast_all, historical_weekly, forecast_weekly = generate_forecast(
                    st.session_state.model_results, train_data, forecast_data_filtered, 
                    selected_features, model_option
                )
                
                # Add day names to the forecast days
                forecast_all['Day_Name'] = forecast_all['Operational Date'].dt.day_name()
                forecast_all['Date_With_Day'] = forecast_all['Day_Name'] + ", " + forecast_all['Operational Date'].dt.strftime('%b %d')
                
                # Add day names to historical days too
                historical_all['Day_Name'] = historical_all['Operational Date'].dt.day_name()
                historical_all['Date_With_Day'] = historical_all['Day_Name'] + ", " + historical_all['Operational Date'].dt.strftime('%b %d')
                
                # Get feature importance
                feature_importance = get_feature_importance(
                    st.session_state.model_results, selected_features, model_option
                )
            
            # Display forecast results
            
            # Forecast visualizations
            forecast_tabs = st.tabs(["📈 Integrated View", "📊 Daily Forecast", "🏷️ Feature Importance"])
            
            with forecast_tabs[0]:
                # Combined historical and forecast view
                st.markdown("<div class='sub-header'>Sales Forecast Time Series</div>", unsafe_allow_html=True)
                
                # Only include historical data if requested
                if include_historical:
                    # Use the Date_With_Day for the x-axis to show day names
                    fig = create_historical_forecast_chart(
                        historical_all, forecast_all,
                        x_col='Operational Date',  # Keep using Operational_Date for the x-axis because it's a datetime
                        historical_y_cols=('Actual_Sales', 'Predicted_Sales'),
                        forecast_y_col='Forecasted_Sales',
                        historical_lower='Lower_Bound' if confidence_interval else None,
                        historical_upper='Upper_Bound' if confidence_interval else None,
                        forecast_lower='Lower_Bound' if confidence_interval else None,
                        forecast_upper='Upper_Bound' if confidence_interval else None,
                        title="Sales Trend: March Historical Data and April Forecast",
                        xlabel="Date",
                        ylabel="Sales ($)"
                    )
                else:
                    # Just show the forecast with day names
                    fig = create_line_chart_with_ci(
                        forecast_all, 'Operational Date', 'Forecasted_Sales', 
                        'Lower_Bound' if confidence_interval else 'Forecasted_Sales', 
                        'Upper_Bound' if confidence_interval else 'Forecasted_Sales',
                        "April Sales Forecast", "Date", "Forecasted Sales ($)"
                    )
                    
                    # Update x-axis tick labels to include day names
                    new_ticktext = forecast_all['Date_With_Day'].tolist()
                    fig.update_xaxes(
                        tickvals=forecast_all['Operational Date'].tolist(),
                        ticktext=new_ticktext
                    )
                
                st.plotly_chart(fig, use_container_width=True, key="integrated_view_chart")
                
                # Summary statistics for both periods in a more compact layout
                col1, col2 = st.columns(2)
                
                with col1:
                    march_total = historical_all['Actual_Sales'].sum()
                    march_avg = historical_all['Actual_Sales'].mean()
                    march_max = historical_all['Actual_Sales'].max()
                    march_max_date_idx = historical_all['Actual_Sales'].idxmax()
                    march_max_date = historical_all.loc[march_max_date_idx, 'Date_With_Day']
                    
                    st.markdown("""
                    <div class="card" style="border-left: 4px solid #4F46E5; background-color: rgba(79, 70, 229, 0.05);">
                        <div style="font-weight: 600; font-size: 1.1rem; color: #4F46E5; margin-bottom: 8px;">March Summary</div>
                        <div style="font-size: 0.8rem; color: #6B7280; margin-bottom: 10px;">Complete month summary</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Total Sales:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Avg. Daily Sales:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Peak Sales:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f} on {}</span>
                        </div>
                    </div>
                    """.format(march_total, march_avg, march_max, march_max_date), unsafe_allow_html=True)
                
                with col2:
                    april_total = forecast_all['Forecasted_Sales'].sum()
                    april_avg = forecast_all['Forecasted_Sales'].mean()
                    april_max = forecast_all['Forecasted_Sales'].max()
                    april_max_date_idx = forecast_all['Forecasted_Sales'].idxmax()
                    april_max_date = forecast_all.loc[april_max_date_idx, 'Date_With_Day']
                    
                    st.markdown("""
                    <div class="card" style="border-left: 4px solid #F59E0B; background-color: rgba(245, 158, 11, 0.05);">
                        <div style="font-weight: 600; font-size: 1.1rem; color: #F59E0B; margin-bottom: 8px;">April Forecast Summary</div>
                        <div style="font-size: 0.8rem; color: #6B7280; margin-bottom: 10px;">First week forecast only</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Total Forecast:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Avg. Daily Forecast:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #1F2937;">Peak Forecast:</span>
                            <span style="font-weight: 600; color: #1F2937;">${:.2f} on {}</span>
                        </div>
                    </div>
                    """.format(april_total, april_avg, april_max, april_max_date), unsafe_allow_html=True)
            
            with forecast_tabs[1]:
                # Daily forecast chart and table with day names
                st.markdown("<div class='sub-header'>Daily Sales Forecast</div>", unsafe_allow_html=True)
                
                # Add March historical data to daily forecast view - use the combined chart
                if include_historical:
                    # Use the same combined chart function from the Integrated View
                    fig = create_historical_forecast_chart(
                        historical_all, forecast_all,
                        x_col='Operational Date',
                        historical_y_cols=('Actual_Sales', 'Predicted_Sales'),
                        forecast_y_col='Forecasted_Sales',
                        historical_lower='Lower_Bound' if confidence_interval else None,
                        historical_upper='Upper_Bound' if confidence_interval else None,
                        forecast_lower='Lower_Bound' if confidence_interval else None,
                        forecast_upper='Upper_Bound' if confidence_interval else None,
                        title="Daily Sales: Historical March Data and April Forecast",
                        xlabel="Date",
                        ylabel="Sales ($)"
                    )
                else:
                    # Create just the daily forecast chart if historical data is not requested
                    fig = create_line_chart_with_ci(
                        forecast_all, 'Operational Date', 'Forecasted_Sales', 
                        'Lower_Bound', 'Upper_Bound', 
                        f"Daily Sales Forecast ({forecast_start.strftime('%b %d')} - {forecast_end.strftime('%b %d')})", 
                        "Date", "Forecasted Sales ($)"
                    )
                    
                    # Update x-axis tick labels to include day names
                    new_ticktext = forecast_all['Date_With_Day'].tolist()
                    fig.update_xaxes(
                        tickvals=forecast_all['Operational Date'].tolist(),
                        ticktext=new_ticktext
                    )
                
                st.plotly_chart(fig, use_container_width=True, key="daily_forecast_chart")
                
                # Daily forecast table with day names
                st.markdown("<div style='margin: 20px 0 10px 0; font-weight: 500; color: var(--text);'>Detailed Daily Forecasts</div>", unsafe_allow_html=True)
                
                daily_display = forecast_all[['Date_With_Day', 'Forecasted_Sales', 'Lower_Bound', 'Upper_Bound']].copy()
                daily_display = daily_display.rename(columns={
                    'Date_With_Day': 'Date',
                    'Forecasted_Sales': 'Forecast ($)',
                    'Lower_Bound': 'Lower Bound ($)',
                    'Upper_Bound': 'Upper Bound ($)'
                })
                
                # Round numeric columns
                for col in ['Forecast ($)', 'Lower Bound ($)', 'Upper Bound ($)']:
                    daily_display[col] = daily_display[col].round(2)
                
                st.dataframe(daily_display, use_container_width=True, hide_index=True, key="daily_forecast_table")
                
                # Add single download button for data
                csv = daily_display.to_csv(index=False)
                
                # Modern styled download button
                st.markdown("""
                <div style="display: flex; justify-content: flex-end; margin-top: 8px;">
                    <div style="background-color: rgba(79, 70, 229, 0.1); border: 1px solid var(--primary); 
                         border-radius: 6px; padding: 4px 10px; display: inline-flex; align-items: center;">
                        <span style="color: var(--primary); font-weight: 500; font-size: 0.9rem;">Download Data</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_download = st.columns([3, 1])
                with col_download[0]:
                    st.write("")  # Placeholder
                    
                with col_download[1]:
                    st.download_button(
                        label="Download Data",
                        data=csv,
                        file_name="daily_sales_forecast.csv",
                        mime="text/csv",
                        key="download_daily_data",
                        help="Download the forecast data as CSV"
                    )
            
            with forecast_tabs[2]:
                # Feature importance visualization
                st.markdown("<div class='sub-header'>Feature Importance Analysis</div>", unsafe_allow_html=True)
                
                # Get feature importance from the model
                feature_importance = get_feature_importance(
                    st.session_state.model_results, selected_features, model_option
                )
                
                # Create a nice feature importance bar chart
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Add a minimum visible value to ensure all bars are visible
                min_importance = feature_importance['Importance'].min()
                visible_threshold = max(min_importance, feature_importance['Importance'].max() * 0.05)
                
                # Create a new column for visualization with minimum bar width
                feature_importance['Visible_Importance'] = feature_importance['Importance'].apply(
                    lambda x: max(x, visible_threshold)
                )
                
                # Create the chart with better visibility
                fig = px.bar(
                    feature_importance,
                    x='Visible_Importance', 
                    y='Feature',
                    orientation='h',
                    title="What's Driving the Predictions?",
                    color='Importance',
                    color_continuous_scale=['#d9ebfd', '#87b3e8', '#4F46E5'],
                    text='Importance'
                )
                
                # Format the text to show actual importance values
                fig.update_traces(
                    texttemplate='%{customdata:.3f}',
                    textposition='outside',
                    customdata=feature_importance['Importance'].values.reshape(-1, 1),
                    marker_line_color='#4F46E5',
                    marker_line_width=1,
                    opacity=0.9
                )
                
                # Customize the chart for our modern design
                colors = get_color_palette()
                
                fig.update_layout(
                    title={
                        'text': "<b>What's Driving the Predictions?</b>",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 18, 'color': colors['text']}
                    },
                    xaxis_title="Relative Importance",
                    yaxis_title="Feature",
                    height=600,  # Increased height for better visibility
                    template="plotly_white",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=150, t=70, b=20),  # Increased right margin for text labels
                    xaxis=dict(
                        showgrid=True,
                        gridcolor=colors['grid'],
                        tickfont=dict(size=12),
                        title_font=dict(size=14),
                        range=[0, feature_importance['Importance'].max() * 1.2]  # Extend x-axis for text labels
                    ),
                    yaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=12),
                        title_font=dict(size=14)
                    ),
                    coloraxis_showscale=False,
                    hoverlabel=dict(
                        bgcolor='white',
                        font_size=12
                    ),
                    font=dict(family="Inter, -apple-system, sans-serif")
                )
                
                # Add value annotations with better visibility
                for i, row in enumerate(feature_importance.itertuples()):
                    feature_name = row.Feature
                    importance_value = row.Importance
                    
                    # Add a clearer annotation for each bar
                    fig.add_annotation(
                        x=row.Visible_Importance,
                        y=feature_name,
                        text=f"{importance_value:.3f}",
                        showarrow=False,
                        xshift=10,
                        font=dict(
                            size=12,
                            color="#4F46E5"
                        ),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#4F46E5",
                        borderwidth=1,
                        borderpad=3,
                        align="left"
                    )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart")
                
                # Add feature importance explanation
                st.markdown("""
                <div class="card" style="border-left: 4px solid #10B981; background-color: rgba(16, 185, 129, 0.05);">
                    <div style="font-weight: 600; font-size: 1.1rem; color: #10B981; margin-bottom: 12px;">Understanding Feature Importance</div>
                    <p style="color: #1F2937; font-size: 0.9rem; margin-bottom: 10px;">
                        The chart above shows which factors have the biggest influence on the sales forecast:
                    </p>
                    <ul style="color: #1F2937; font-size: 0.9rem; margin-left: 20px; margin-bottom: 10px;">
                        <li><b>Higher values</b> indicate the feature has a <b>stronger influence</b> on predictions</li>
                        <li>These insights can help you understand what drives sales patterns</li>
                        <li>Consider focusing on high-impact factors for business planning</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Add feature descriptions
                st.markdown("<div style='margin: 20px 0 10px 0; font-weight: 500; color: var(--text);'>Feature Glossary</div>", unsafe_allow_html=True)
                
                # Create a dataframe to show feature descriptions
                feature_descriptions = {
                    'dayofweek': 'Day of week (0-6, 0=Monday)',
                    'dayofmonth': 'Day of month (1-31)',
                    'week': 'Week number of the year',
                    'week_of_month': 'Week of the month (1-5)',
                    'Is_Weekend': 'Whether the day is a weekend (1) or weekday (0)',
                    'Is_Closed': 'Whether the day is closed (1) or open (0)',
                    'tempmax': 'Maximum daily temperature',
                    'tempmin': 'Minimum daily temperature',
                    'temp': 'Average daily temperature',
                    'humidity': 'Average humidity percentage',
                    'precip': 'Precipitation amount',
                    'precipprob': 'Probability of precipitation',
                    'cloudcover': 'Percentage of cloud cover',
                    'solarradiation': 'Solar radiation level',
                    'uvindex': 'UV index'
                }
                
                # Filter to only show descriptions for features actually used
                used_features = feature_importance['Feature'].tolist()
                used_descriptions = {feature: desc for feature, desc in feature_descriptions.items() if feature in used_features}
                
                # Create the glossary dataframe
                glossary_df = pd.DataFrame({
                    'Feature': used_descriptions.keys(),
                    'Description': used_descriptions.values()
                })
                
                # Display the glossary
                st.dataframe(glossary_df, use_container_width=True, hide_index=True)
        else:
            # Show placeholder when no forecast is generated
            st.markdown("""
            <div style="text-align: center; padding: 40px; background-color: rgba(79, 70, 229, 0.03); 
                 border-radius: 8px; border: 1px dashed rgba(79, 70, 229, 0.3); margin: 20px 0;">
                <div style="font-size: 3rem; margin-bottom: 10px; color: #818CF8;">📊</div>
                <h3 style="color: #4F46E5; font-weight: 600; margin-bottom: 16px;">Forecast Preview</h3>
                <p style="color: #6B7280; max-width: 500px; margin: 0 auto 20px auto;">
                    Configure your forecast settings and click 'Generate Forecast' to see your sales analysis.
                </p>
                <div style="width: 100px; height: 4px; background-color: rgba(79, 70, 229, 0.2); 
                     margin: 0 auto; border-radius: 2px;"></div>
            </div>
            """, unsafe_allow_html=True)
    
    # Remove the footnote
    # st.markdown("<div class='footnote'>Forecast generated using time series analysis of March historical data to predict April sales. Confidence intervals based on bootstrap sampling.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 