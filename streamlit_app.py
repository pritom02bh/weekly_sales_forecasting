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
import warnings
import hashlib
import time
from typing import Dict, List, Tuple, Optional

# Set page config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Suppress cache warnings
warnings.filterwarnings('ignore')

# Hide cache warnings in UI
st.markdown("""
<style>
    .stAlert[data-baseweb="notification"] {
        display: none !important;
    }
    .stWarning {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache optimization with TTL
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_file_size(file_path: str) -> int:
    """Get file size in bytes with caching"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

@st.cache_data(ttl=3600)
def list_model_files() -> List[str]:
    """Cache list of model files to avoid repeated filesystem operations"""
    model_dir = get_model_path("")
    if not os.path.exists(model_dir):
        return []
    return [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

# Function to resolve paths for data files
def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), "data", filename)

# Function to resolve paths for model files
def get_model_path(filename):
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return os.path.join(model_dir, filename)

# Function to generate model filename
def generate_model_filename(company_id, model_name, features_hash):
    """Generate a unique filename for the model based on company, model type, and features"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"model_{company_id}_{model_name.lower().replace(' ', '_')}_{features_hash[:8]}_{timestamp}.pkl"

# Function to get features hash for model versioning
@st.cache_data
def get_features_hash(features):
    """Generate a hash of the features list for model versioning"""
    features_str = ''.join(sorted(features))
    return hashlib.md5(features_str.encode()).hexdigest()

# Optimized model saving with compression
def save_model_with_metadata(model, model_name, company_id, features, metrics, filename):
    """Save model with comprehensive metadata and compression"""
    try:
        model_data = {
            'model': model,
            'model_name': model_name,
            'company_id': company_id,
            'features': features,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'streamlit_version': st.__version__,
            'model_version': '1.0'
        }
        
        model_path = get_model_path(filename)
        # Use compression to reduce file size
        joblib.dump(model_data, model_path, compress=3)
        
        return model_path
        
    except Exception as e:
        st.error(f"❌ Error saving model: {str(e)}")
        return None

# Optimized model loading with better error handling
def load_model_with_metadata(filename):
    """Load model with validation and error handling"""
    try:
        model_path = get_model_path(filename)
        
        if not os.path.exists(model_path):
            return None
            
        model_data = joblib.load(model_path)
        
        # Validate required fields
        required_fields = ['model', 'model_name', 'company_id', 'features', 'metrics', 'timestamp']
        if not all(field in model_data for field in required_fields):
            st.warning(f"⚠️ Model file {filename} is missing required metadata")
            return None
            
        return model_data
        
    except Exception as e:
        st.error(f"❌ Error loading model {filename}: {str(e)}")
        return None

# Optimized existing models finder
def find_existing_models(company_id, features):
    """Find existing models for a company with matching features - optimized"""
    try:
        model_files = list_model_files()
        if not model_files:
            return []
            
        features_hash = get_features_hash(features)
        existing_models = []
        
        for filename in model_files:
            if filename.startswith(f"model_{company_id}_") and filename.endswith('.pkl'):
                model_data = load_model_with_metadata(filename)
                if model_data and model_data['features'] == features:
                    model_info = {
                        'filename': filename,
                        'model_name': model_data['model_name'],
                        'timestamp': model_data['timestamp'],
                        'metrics': model_data['metrics'],
                        'model_data': model_data
                    }
                    existing_models.append(model_info)
        
        # Sort by timestamp (newest first)
        existing_models.sort(key=lambda x: x['timestamp'], reverse=True)
        return existing_models
        
    except Exception as e:
        st.error(f"❌ Error finding existing models: {str(e)}")
        return []

# Function to cleanup old models - optimized
def cleanup_old_models(company_id, days_threshold=7):
    """Remove models older than specified days"""
    try:
        model_files = list_model_files()
        if not model_files:
            return 0
            
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        removed_count = 0
        
        for filename in model_files:
            if filename.startswith(f"model_{company_id}_"):
                model_data = load_model_with_metadata(filename)
                if model_data:
                    model_timestamp = pd.to_datetime(model_data['timestamp']).strftime('%Y-%m-%d')
                    if model_timestamp < cutoff_date:
                        model_path = get_model_path(filename)
                        os.remove(model_path)
                        removed_count += 1
        
        # Clear caches to reflect changes
        list_model_files.clear()
        
        return removed_count
        
    except Exception as e:
        st.error(f"❌ Error cleaning up models: {str(e)}")
        return 0

# Optimized storage info
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_model_storage_info():
    """Get information about model storage usage - cached"""
    try:
        model_files = list_model_files()
        if not model_files:
            return {"total_models": 0, "total_size_mb": 0}
            
        total_models = len(model_files)
        total_size = sum(get_file_size(get_model_path(f)) for f in model_files)
        
        return {
            "total_models": total_models,
            "total_size_mb": total_size / (1024 * 1024)
        }
        
    except Exception:
        return {"total_models": 0, "total_size_mb": 0}

# Add CSS for better UI performance
st.markdown("""
<style>
    /* Remove ALL default Streamlit padding and margins */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.3rem !important;
        padding-right: 0.3rem !important;
        max-width: 100% !important;
        margin-top: 0rem !important;
    }
    
    /* Remove Streamlit header space completely */
    .stApp > header {
        height: 0rem !important;
        display: none !important;
    }
    
    /* Remove ALL top spacing */
    .stApp {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Remove any default container spacing */
    .stApp > div:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
    
    /* Ultra-compact header */
    .clean-header {
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        color: white;
        padding: 0.15rem 0.4rem !important;
        border-radius: 0.2rem;
        margin: 0 0 0.3rem 0 !important;
        font-size: 0.75rem !important;
    }
    
    .clean-header-content {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    .clean-header-icon {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.15rem;
        border-radius: 0.1rem;
        font-size: 0.7rem;
    }
    
    .clean-header-title {
        margin: 0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        line-height: 1 !important;
    }
    
    /* Super compact sections */
    .compact-section {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.2rem;
        padding: 0.4rem 0.5rem;
        margin: 0.2rem 0;
    }
    
    .compact-section h3 {
        margin: 0 0 0.3rem 0 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #495057 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Ultra-compact metric cards */
    .metric-card-compact {
        background: #ffffff;
        padding: 0.3rem;
        border-radius: 0.2rem;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 0.15rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .metric-value-compact {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label-compact {
        font-size: 0.6rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin: 0;
    }
    
    /* Compact info boxes */
    .info-box-compact {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.15rem;
        padding: 0.3rem 0.4rem;
        margin: 0.15rem 0;
        font-size: 0.7rem;
        line-height: 1.2;
    }
    
    .info-box-compact strong {
        color: #495057;
        font-size: 0.65rem;
    }
    
    /* Compact form elements */
    .stSelectbox > div > div {
        background-color: white;
        margin: 0.1rem 0;
        min-height: 2rem !important;
    }
    
    .stSelectbox label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.1rem !important;
    }
    
    .stDateInput > div > div {
        margin: 0.1rem 0;
    }
    
    .stDateInput label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.1rem !important;
    }
    
    .stCheckbox {
        margin: 0.1rem 0;
    }
    
    .stCheckbox label {
        font-size: 0.7rem !important;
    }
    
    .stButton > button {
        padding: 0.2rem 0.4rem;
        margin: 0.1rem 0;
        font-size: 0.7rem;
        border-radius: 0.2rem;
        min-height: 2rem !important;
    }
    
    .stButton[data-testid="primary"] > button {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        border: none;
        color: white;
        font-weight: 600;
    }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        padding: 0.2rem 0.5rem;
        font-size: 0.7rem;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    .streamlit-expanderContent {
        padding: 0.3rem 0.5rem;
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-top: none;
    }
    
    /* Remove excessive spacing from various elements */
    .element-container {
        margin: 0.1rem 0 !important;
    }
    
    /* Compact columns */
    .stColumns > div {
        padding: 0 0.1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {visibility: hidden !important;}
    
    /* Make the left column narrower */
    .stColumns > div:first-child {
        min-width: 200px !important;
        max-width: 220px !important;
    }
    
    /* Compact slider */
    .stSlider {
        margin: 0.1rem 0;
    }
    
    .stSlider label {
        font-size: 0.7rem !important;
    }
    
    /* Compact color picker */
    .stColorPicker {
        margin: 0.1rem 0;
    }
    
    .stColorPicker label {
        font-size: 0.7rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Company mapping - consolidated and corrected
COMPANY_INFO = {
    "All Locations Combined": {
        "file": "All_Locations_Combined.csv",
        "id": "combined",
        "name": "All Locations Combined",
        "weather_file": "Weather_May15_to_May27.csv"
    },
    "Fenix Food Factory (50460)": {
        "file": "Fenix_Food_Factory_B.V._50460.csv", 
        "id": "50460",
        "name": "Fenix Food Factory B.V.",
        "weather_file": "Weather_May15_to_May27.csv"
    },
    "Kaapse Maria (47903)": {
        "file": "Kaapse_Maria_B.V._47903.csv",
        "id": "47903", 
        "name": "Kaapse Maria B.V.",
        "weather_file": "Weather_May15_to_May27.csv"
    },
    "Kaapse Will'ns (47904)": {
        "file": "Kaapse_Will'ns_B.V._47904.csv",
        "id": "47904",
        "name": "Kaapse Will'ns B.V.", 
        "weather_file": "Weather_May15_to_May27.csv"
    },
    "Kaapse Kaap (47901)": {
        "file": "Kaapse_Kaap_B.V._47901.csv",
        "id": "47901",
        "name": "Kaapse Kaap B.V.",
        "weather_file": "Weather_Kaapse_Kaap_47901.csv"  # Using location-specific weather as requested
    }
}

# Add data validation flags to track weather data usage
WEATHER_DATA_STATUS = {
    "47901": "Using actual weather data: Dec 5, 2024 - Jan 4, 2025 (31 days)",
    "47903": "Using shared forecast weather: Weather_May15_to_May27.csv",
    "47904": "Using shared forecast weather: Weather_May15_to_May27.csv", 
    "50460": "Using shared forecast weather: Weather_May15_to_May27.csv",
    "combined": "Using shared forecast weather: Weather_May15_to_May27.csv"
}

# Define company configurations
company_configs = {
    "Kaapse_Kaap_B.V._47901": {
        "name": "Kaapse Kaap B.V.",
        "id": "47901",
        "file": "Kaapse_Kaap_B.V._47901.csv",
        "weather_file": "Weather_Kaapse_Kaap_47901.csv",  # Correct individual file
        "location": "Kaapse Kaap"
    },
    "Kaapse_Maria_B.V._47903": {
        "name": "Kaapse Maria B.V.",
        "id": "47903",
        "file": "Kaapse_Maria_B.V._47903.csv",
        "weather_file": "Weather_May15_to_May27.csv",  # Use as fallback
        "location": "Kaapse Maria"
    },
    "Grand_Total_C.V._47905": {
        "name": "Grand Total C.V.",
        "id": "47905",
        "file": "Grand_Total_C.V._47905.csv",
        "weather_file": "Weather_May15_to_May27.csv",
        "location": "Grand Total"
    },
    "Fenix_Food_Factory_B.V._50460": {
        "name": "Fenix Food Factory B.V.",
        "id": "50460",
        "file": "Fenix_Food_Factory_B.V._50460.csv",
        "weather_file": "Weather_May15_to_May27.csv",
        "location": "Fenix Food Factory"
    },
    "All_Locations_Combined": {
        "name": "All Locations Combined",
        "id": "combined",
        "file": "All_Locations_Combined.csv",
        "weather_file": "Weather_May15_to_May27.csv",
        "location": "All Locations"
    }
}

# Optimized data loading with enhanced validation
@st.cache_data
def load_data(selected_company):
    """Load data for the selected company with comprehensive error handling and validation"""
    try:
        company_config = COMPANY_INFO[selected_company]
        errors = []
        weather_data = None
        
        # Load sales data with validation
        try:
            sales_file_path = get_data_path(company_config["file"])
            if not os.path.exists(sales_file_path):
                raise FileNotFoundError(f"Sales data file not found: {company_config['file']}")
            
            sales_data = pd.read_csv(sales_file_path)
            
            if len(sales_data) == 0:
                raise ValueError(f"Sales data file {company_config['file']} is empty")
                
            # Validate required columns
            required_columns = ['Operational Date', 'Total_Sales']
            missing_columns = [col for col in required_columns if col not in sales_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in sales data: {missing_columns}")
            
            # Additional validation for individual companies
            if len(sales_data) < 30:
                # st.info(f"ℹ️ **{company_config['name']}** has limited historical data ({len(sales_data)} records). Forecasts may be less precise for individual locations.")
                pass
            
            # Check for data quality issues specific to individual companies
            zero_sales_ratio = (sales_data['Total_Sales'] == 0).sum() / len(sales_data)
            if zero_sales_ratio > 0.5:
                # st.warning(f"⚠️ **{company_config['name']}** has many zero-sales days ({zero_sales_ratio:.1%}). This may indicate frequent closures or data quality issues.")
                pass
        
        except Exception as e:
            errors.append(f"Sales data loading error: {str(e)}")
            raise
        
        # Load weather data with enhanced validation
        try:
            weather_file_path = get_data_path(company_config["weather_file"])
            if os.path.exists(weather_file_path):
                weather_data = pd.read_csv(weather_file_path)
                if len(weather_data) == 0:
                    weather_data = None
            else:
                weather_data = None
        except Exception as e:
            weather_data = None
        
        # Create default weather data if needed
        if weather_data is None:
            # Create weather data covering the forecast period
            default_start = pd.Timestamp('2025-05-15')
            default_end = pd.Timestamp('2025-05-27')
            default_dates = pd.date_range(start=default_start, end=default_end, freq='D')
            
            weather_data = pd.DataFrame({
                'Operational Date': default_dates,
                'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
                'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
                'solarradiation': 200, 'uvindex': 6
            })
        
        # Smart date parsing with detailed validation
        sales_data = parse_dates_intelligently(sales_data, 'sales', company_config)
        if weather_data is not None:
            weather_data = parse_dates_intelligently(weather_data, 'weather', company_config)
        
        # Data quality validation and cleanup
        sales_data = validate_and_clean_sales_data(sales_data)
        weather_data = validate_and_clean_weather_data(weather_data)
        
        # Enhanced weather data validation
        weather_valid, weather_msg = validate_weather_data_usage(company_config, weather_data)
        merge_valid, merge_msg = validate_sales_weather_merge(sales_data, weather_data)
        
        # Special info for Kaapse Kaap enhanced forecasting
        if company_config["id"] == "47901":
            # st.info("ℹ️ **Kaapse Kaap Forecasting**: Using actual weather data for December 5, 2024 - January 4, 2025 (31 days).")
            pass
        
        # Only show warnings for actual issues, not expected forecast behavior
        if not weather_valid:
            st.warning(f"⚠️ Weather data issue: {weather_msg}")
        if not merge_valid:
            st.warning(f"⚠️ Data alignment issue: {merge_msg}")
        
        # Final validation
        if len(sales_data) == 0:
            raise ValueError("No valid sales data remaining after cleaning")
        
        return sales_data, weather_data, company_config
        
    except Exception as e:
        for error in errors:
            st.error(f"❌ {error}")
        st.error(f"❌ Critical error loading data for {selected_company}: {str(e)}")
        st.stop()

def parse_dates_intelligently(data, data_type, company_config):
    """Intelligent date parsing with multiple fallback strategies"""
    if data is None or len(data) == 0:
        return data
    
    try:
        # Company-specific format mapping with validation
        company_date_formats = {
            "combined": "DD-MM-YYYY",      # All Locations Combined
            "50460": "DD-MM-YYYY",         # Fenix Food Factory  
            "47903": "DD-MM-YYYY",         # Kaapse Maria
            "47904": "DD-MM-YYYY",         # Kaapse Will'ns
            "47901": "YYYY-MM-DD"          # Kaapse Kaap (different format!)
        }
        
        # Weather data is typically in YYYY-MM-DD format
        if data_type == 'weather':
            expected_format = "YYYY-MM-DD"
        else:
            expected_format = company_date_formats.get(company_config["id"], "auto")
        
        # Sample a few dates for format detection
        original_dates = data['Operational Date'].dropna().astype(str).head(10)
        if len(original_dates) == 0:
            raise ValueError(f"No dates found in {data_type} data")
        
        # Multiple parsing strategies
        parsing_strategies = []
        
        if expected_format == "DD-MM-YYYY":
            parsing_strategies = [
                ('%d-%m-%Y', 'DD-MM-YYYY'),
                ('%Y-%m-%d', 'YYYY-MM-DD'), 
                ('auto', 'Auto-detect')
            ]
        elif expected_format == "YYYY-MM-DD":
            parsing_strategies = [
                ('%Y-%m-%d', 'YYYY-MM-DD'),
                ('%d-%m-%Y', 'DD-MM-YYYY'),
                ('auto', 'Auto-detect')
            ]
        else:
            parsing_strategies = [
                ('%d-%m-%Y', 'DD-MM-YYYY'),
                ('%Y-%m-%d', 'YYYY-MM-DD'),
                ('auto', 'Auto-detect')
            ]
        
        success = False
        best_result = None
        best_format = None
        best_valid_count = 0
        
        for date_format, format_name in parsing_strategies:
            try:
                data_copy = data.copy()
                
                if date_format == 'auto':
                    # Auto-detect parsing
                    data_copy['Operational Date'] = pd.to_datetime(data_copy['Operational Date'], errors='coerce')
                else:
                    # Specific format parsing
                    data_copy['Operational Date'] = pd.to_datetime(data_copy['Operational Date'], 
                                                                   format=date_format, errors='coerce')
                
                # Count valid dates
                valid_count = data_copy['Operational Date'].notna().sum()
                
                # Check if this parsing is better
                if valid_count > best_valid_count:
                    best_valid_count = valid_count
                    best_result = data_copy
                    best_format = format_name
                    
                    # If we got >80% success rate, that's probably the right format
                    if valid_count / len(data_copy) > 0.8:
                        success = True
                        break
                        
            except Exception:
                continue
        
        # Use the best result if we found one
        if best_result is not None and best_valid_count > 0:
            data = best_result
            
            # Drop rows with invalid dates
            original_len = len(data)
            data = data.dropna(subset=['Operational Date'])
            dropped_count = original_len - len(data)
            
            if dropped_count > 0:
                drop_percentage = (dropped_count / original_len) * 100
                if drop_percentage > 20:  # Only warn if significant data loss
                    # Remove technical warning - users don't need to see parsing details
                    pass
            
            # Show successful parsing info (only for significant datasets) - REMOVED FOR CLEANER UI
            # Remove technical parsing details that users don't need to see
            pass
        
        else:
            raise ValueError(f"Could not parse any dates in {data_type} data using any known format")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Date parsing failed for {data_type} data: {str(e)}")
        raise

def validate_and_clean_sales_data(sales_data):
    """Comprehensive sales data validation and cleaning with missing date handling"""
    try:
        original_len = len(sales_data)
        
        # **CRITICAL FIX**: Remove completely empty rows first
        sales_data = sales_data.dropna(how='all')
        
        # Remove rows with missing critical data
        sales_data = sales_data.dropna(subset=['Operational Date', 'Total_Sales'])
        
        # **NEW**: Filter out rows with empty or invalid sales data
        sales_data = sales_data[sales_data['Total_Sales'].notna()]
        sales_data = sales_data[sales_data['Total_Sales'] != '']
        
        # Remove negative sales (data quality issue)
        sales_data = sales_data[sales_data['Total_Sales'] >= 0]
        
        # **MISSING DATE FIX**: Fill ALL missing dates as closed days (zero sales)
        # Sort by date to identify gaps
        sales_data = sales_data.sort_values('Operational Date')
        
        # Check for date gaps and ALWAYS fill them as closed days
        if len(sales_data) > 1:
            # Get the full date range from first to last date
            min_date = sales_data['Operational Date'].min()
            max_date = sales_data['Operational Date'].max()
            
            # Create complete date range
            complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create a DataFrame with all dates
            complete_df = pd.DataFrame({'Operational Date': complete_date_range})
            
            # Merge with existing data, filling missing values
            sales_data = pd.merge(complete_df, sales_data, on='Operational Date', how='left')
            
            # Fill missing sales data with zeros (closed days)
            sales_data['Total_Sales'] = sales_data['Total_Sales'].fillna(0)
            sales_data['Sales_Count'] = sales_data['Sales_Count'].fillna(0)
            
            # Mark missing days as closed
            if 'Is_Closed' not in sales_data.columns:
                sales_data['Is_Closed'] = 0
            
            # Set Is_Closed = 1 for days with zero sales (including filled missing dates)
            sales_data.loc[sales_data['Total_Sales'] == 0, 'Is_Closed'] = 1
            
            # Fill other categorical columns with appropriate defaults
            categorical_fills = {
                'Day_of_Week': sales_data['Operational Date'].dt.day_name(),
                'Is_Weekend': (sales_data['Operational Date'].dt.dayofweek >= 5).astype(int),
                'Is_Public_Holiday': 0,
                'Tips_per_Transaction': 0,
                'Avg_Sale_per_Transaction': 0
            }
            
            for col, fill_values in categorical_fills.items():
                if col in sales_data.columns:
                    if col == 'Day_of_Week':
                        sales_data[col] = sales_data[col].fillna(fill_values)
                    else:
                        sales_data[col] = sales_data[col].fillna(fill_values)
            
            # Fill weather columns with interpolation or forward fill for missing dates
            weather_columns = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'cloudcover', 'solarradiation', 'uvindex']
            for col in weather_columns:
                if col in sales_data.columns:
                    # Try interpolation first, then forward fill, then backward fill
                    sales_data[col] = sales_data[col].interpolate(method='linear', limit_direction='both')
                    sales_data[col] = sales_data[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still missing, use reasonable defaults
                    if sales_data[col].isna().any():
                        weather_defaults = {
                            'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
                            'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
                            'solarradiation': 200, 'uvindex': 6
                        }
                        sales_data[col] = sales_data[col].fillna(weather_defaults.get(col, 0))
        
        # Basic data quality checks
        if sales_data['Total_Sales'].min() < 0:
            # st.warning("⚠️ Found negative sales values - cleaning data")
            sales_data = sales_data[sales_data['Total_Sales'] >= 0]
        
        if 'Sales_Count' in sales_data.columns and sales_data['Sales_Count'].min() < 0:
            # st.warning("⚠️ Found negative transaction counts - cleaning data")
            sales_data = sales_data[sales_data['Sales_Count'] >= 0]
        
        # Remove outliers (sales more than 5 standard deviations from mean) - but only for open days
        open_days_sales = sales_data[sales_data['Total_Sales'] > 0]['Total_Sales']
        if len(open_days_sales) > 0:
            mean_sales = open_days_sales.mean()
            std_sales = open_days_sales.std()
            outlier_threshold = mean_sales + 5 * std_sales
            
            outliers = (sales_data['Total_Sales'] > outlier_threshold) & (sales_data['Total_Sales'] > 0)
            if outliers.any():
                # st.warning(f"⚠️ Removing {outliers.sum()} extreme outliers from the data")
                sales_data = sales_data[~outliers]
        
        # Data type conversions
        sales_data['Total_Sales'] = pd.to_numeric(sales_data['Total_Sales'], errors='coerce')
        if 'Sales_Count' in sales_data.columns:
            sales_data['Sales_Count'] = pd.to_numeric(sales_data['Sales_Count'], errors='coerce')
        
        # Fill any remaining NaN values with appropriate defaults
        if 'Is_Weekend' in sales_data.columns:
            sales_data['Is_Weekend'] = sales_data['Is_Weekend'].fillna(0)
        if 'Is_Public_Holiday' in sales_data.columns:
            sales_data['Is_Public_Holiday'] = sales_data['Is_Public_Holiday'].fillna(0)
        if 'Is_Closed' in sales_data.columns:
            sales_data['Is_Closed'] = sales_data['Is_Closed'].fillna(0)
        
        # Weather data cleaning
        weather_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip']
        for col in weather_cols:
            if col in sales_data.columns:
                sales_data[col] = pd.to_numeric(sales_data[col], errors='coerce')
                sales_data[col] = sales_data[col].fillna(sales_data[col].median())
        
        # Sort by date again
        sales_data = sales_data.sort_values('Operational Date').reset_index(drop=True)
        
        cleaned_len = len(sales_data)
        
        if cleaned_len > original_len:
            filled_dates = cleaned_len - original_len
            # st.info(f"📅 Filled {filled_dates} missing dates as closed days (will show as red dots)")
        elif cleaned_len < original_len:
            # st.info(f"📊 Data cleaned: {original_len} → {cleaned_len} rows ({original_len - cleaned_len} rows removed)")
            pass
        
        return sales_data
        
    except Exception as e:
        st.error(f"❌ Error during data validation: {str(e)}")
        return sales_data

def fill_missing_dates_as_closed(sales_data):
    """Fill all missing dates in the data range as closed days (zero sales)"""
    try:
        if len(sales_data) == 0:
            return sales_data
        
        # Get the full date range
        min_date = sales_data['Operational Date'].min()
        max_date = sales_data['Operational Date'].max()
        
        # Create complete date range
        complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create a DataFrame with all dates
        complete_df = pd.DataFrame({'Operational Date': complete_date_range})
        
        # Merge with existing data, filling missing values
        sales_data = pd.merge(complete_df, sales_data, on='Operational Date', how='left')
        
        # Fill missing sales data with zeros (closed days)
        sales_data['Total_Sales'] = sales_data['Total_Sales'].fillna(0)
        sales_data['Sales_Count'] = sales_data['Sales_Count'].fillna(0)
        
        # Mark missing days as closed
        if 'Is_Closed' not in sales_data.columns:
            sales_data['Is_Closed'] = 0
        
        # Set Is_Closed = 1 for days with zero sales
        sales_data.loc[sales_data['Total_Sales'] == 0, 'Is_Closed'] = 1
        
        # Fill other categorical columns with appropriate defaults
        categorical_fills = {
            'Day_of_Week': sales_data['Operational Date'].dt.day_name(),
            'Is_Weekend': (sales_data['Operational Date'].dt.dayofweek >= 5).astype(int),
            'Is_Public_Holiday': 0,
            'Tips_per_Transaction': 0,
            'Avg_Sale_per_Transaction': 0
        }
        
        for col, fill_values in categorical_fills.items():
            if col in sales_data.columns:
                if col == 'Day_of_Week':
                    sales_data[col] = sales_data[col].fillna(fill_values)
                else:
                    sales_data[col] = sales_data[col].fillna(fill_values)
        
        # Fill weather columns with interpolation or forward fill for missing dates
        weather_columns = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'cloudcover', 'solarradiation', 'uvindex']
        for col in weather_columns:
            if col in sales_data.columns:
                # Try interpolation first, then forward fill, then backward fill
                sales_data[col] = sales_data[col].interpolate(method='linear', limit_direction='both')
                sales_data[col] = sales_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use reasonable defaults
                if sales_data[col].isna().any():
                    weather_defaults = {
                        'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
                        'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
                        'solarradiation': 200, 'uvindex': 6
                    }
                    sales_data[col] = sales_data[col].fillna(weather_defaults.get(col, 0))
        
        # Sort by date
        sales_data = sales_data.sort_values('Operational Date').reset_index(drop=True)
        
        missing_dates_filled = len(complete_date_range) - len(sales_data.dropna(subset=['Total_Sales']))
        if missing_dates_filled > 0:
            pass  # Remove UI clutter - don't show detailed missing dates info
        
        return sales_data
        
    except Exception as e:
        st.error(f"❌ Error filling missing dates: {str(e)}")
        return sales_data

def validate_and_clean_weather_data(weather_data):
    """Weather data validation and cleaning"""
    if weather_data is None:
        return None
        
    try:
        original_len = len(weather_data)
        
        # Remove rows with missing dates
        weather_data = weather_data.dropna(subset=['Operational Date'])
        
        # Fill missing weather values with reasonable defaults
        weather_defaults = {
            'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
            'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
            'solarradiation': 200, 'uvindex': 6
        }
        
        for col, default_val in weather_defaults.items():
            if col in weather_data.columns:
                missing_count = weather_data[col].isna().sum()
                if missing_count > 0:
                    weather_data[col] = weather_data[col].fillna(default_val)
                    if missing_count / len(weather_data) > 0.2:
                        # Remove technical info - users don't need weather filling details
                        pass
        
        # Sort by date
        weather_data = weather_data.sort_values('Operational Date').reset_index(drop=True)
        
        cleaned_count = len(weather_data)
        if original_len - cleaned_count > 0:
            # Remove technical info - users don't need weather cleaning details
            pass
        
        return weather_data
        
    except Exception as e:
        st.error(f"❌ Weather data validation failed: {str(e)}")
        return None

def display_data_summary(sales_data, weather_data, company_config):
    """Display comprehensive data summary"""
    try:
        # Sales data summary
        sales_min_date = sales_data['Operational Date'].min()
        sales_max_date = sales_data['Operational Date'].max()
        total_sales = sales_data['Total_Sales'].sum()
        avg_daily_sales = sales_data['Total_Sales'].mean()
        
        # Date range and coverage
        date_range_days = (sales_max_date - sales_min_date).days + 1
        data_coverage = (len(sales_data) / date_range_days) * 100 if date_range_days > 0 else 0
        
        # Weather data summary
        weather_info = "None"
        if weather_data is not None and len(weather_data) > 0:
            weather_min_date = weather_data['Operational Date'].min()
            weather_max_date = weather_data['Operational Date'].max()
            weather_info = f"{len(weather_data)} records ({weather_min_date.strftime('%Y-%m-%d')} to {weather_max_date.strftime('%Y-%m-%d')})"
        
        # Data quality indicators
        zero_sales_days = len(sales_data[sales_data['Total_Sales'] == 0])
        closure_info = ""
        if 'Is_Closed' in sales_data.columns:
            closed_days = sales_data['Is_Closed'].sum()
            closure_info = f" | {closed_days} marked as closed"
        
        st.markdown(f"""
        <div class="info-box-clean">
            <h4>{company_config['name']}</h4>
            <p>
                <strong>📊 Sales Data:</strong> {len(sales_data):,} records | {sales_min_date.strftime('%Y-%m-%d')} to {sales_max_date.strftime('%Y-%m-%d')}<br>
                <strong>💰 Total Sales:</strong> ${total_sales:,.0f} | <strong>Daily Avg:</strong> ${avg_daily_sales:,.0f}<br>
                <strong>📅 Coverage:</strong> {data_coverage:.1f}% of date range | {zero_sales_days} zero-sales days{closure_info}<br>
                <strong>🌤️ Weather:</strong> {weather_info}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show warnings for data quality issues
        if data_coverage < 70:
            st.warning(f"⚠️ Low data coverage ({data_coverage:.1f}%) - many missing dates in the range")
        
        if len(sales_data) < 30:
            st.warning(f"⚠️ Limited data: Only {len(sales_data)} records. Model predictions may be less reliable.")
        
        if company_config["id"] == "47901" and len(sales_data) < 100:
            st.info("ℹ️ Kaapse Kaap has limited historical data. Consider using Combined Locations for more reliable forecasting.")
        
    except Exception as e:
        st.error(f"❌ Error displaying data summary: {str(e)}")

# Optimized feature engineering with better caching
@st.cache_data
def engineer_features(sales_data, weather_data, company_config):
    """Engineer features for the selected company data - with proper historical/forecast weather handling"""
    
    # Create copies
    train_data = sales_data.copy()
    forecast_data = weather_data.copy()
    
    # HISTORICAL DATA PROCESSING (sales_data already contains embedded weather)
    # Extract temporal features for training data
    train_data['dayofweek'] = train_data['Operational Date'].dt.dayofweek
    train_data['dayofmonth'] = train_data['Operational Date'].dt.day
    train_data['week'] = train_data['Operational Date'].dt.isocalendar().week
    train_data['month'] = train_data['Operational Date'].dt.month
    train_data['quarter'] = train_data['Operational Date'].dt.quarter
    train_data['year'] = train_data['Operational Date'].dt.year
    
    # Rolling averages for sales trends (shorter window for responsiveness)
    train_data['sales_rolling_7'] = train_data['Total_Sales'].rolling(window=7, min_periods=1).mean()
    train_data['sales_rolling_14'] = train_data['Total_Sales'].rolling(window=14, min_periods=1).mean()
    train_data['sales_rolling_30'] = train_data['Total_Sales'].rolling(window=30, min_periods=1).mean()
    
    # Lag features (previous sales patterns)
    train_data['sales_lag_1'] = train_data['Total_Sales'].shift(1)
    train_data['sales_lag_7'] = train_data['Total_Sales'].shift(7)
    train_data['sales_lag_14'] = train_data['Total_Sales'].shift(14)
    
    # Seasonal patterns (weekday vs weekend impact)
    train_data['is_monday'] = (train_data['dayofweek'] == 0).astype(int)
    train_data['is_friday'] = (train_data['dayofweek'] == 4).astype(int)
    train_data['is_saturday'] = (train_data['dayofweek'] == 5).astype(int)
    train_data['is_sunday'] = (train_data['dayofweek'] == 6).astype(int)
    
    # Month seasonality
    train_data['is_month_start'] = (train_data['dayofmonth'] <= 5).astype(int)
    train_data['is_month_end'] = (train_data['dayofmonth'] >= 25).astype(int)
    
    # Weather interaction features
    train_data['temp_range'] = train_data['tempmax'] - train_data['tempmin']
    train_data['comfortable_weather'] = ((train_data['temp'] >= 60) & (train_data['temp'] <= 75) & (train_data['precip'] <= 0.1)).astype(int)
    train_data['bad_weather'] = ((train_data['temp'] <= 40) | (train_data['temp'] >= 85) | (train_data['precip'] >= 0.5)).astype(int)
    
    # FORECAST DATA PROCESSING (weather_data for future dates)
    # Add temporal features to forecast data  
    forecast_data['dayofweek'] = forecast_data['Operational Date'].dt.dayofweek
    forecast_data['dayofmonth'] = forecast_data['Operational Date'].dt.day
    forecast_data['week'] = forecast_data['Operational Date'].dt.isocalendar().week
    forecast_data['month'] = forecast_data['Operational Date'].dt.month
    forecast_data['quarter'] = forecast_data['Operational Date'].dt.quarter
    forecast_data['year'] = forecast_data['Operational Date'].dt.year
    
    # Create weekend/holiday indicators for forecast data
    forecast_data['Is_Weekend'] = forecast_data['dayofweek'].isin([5, 6]).astype(int)
    forecast_data['Is_Public_Holiday'] = 0  # Assume no holidays in forecast period
    forecast_data['Is_Closed'] = 0  # Assume open for forecast days
    
    # Weekday indicators for forecast data
    forecast_data['is_monday'] = (forecast_data['dayofweek'] == 0).astype(int)
    forecast_data['is_friday'] = (forecast_data['dayofweek'] == 4).astype(int)
    forecast_data['is_saturday'] = (forecast_data['dayofweek'] == 5).astype(int)
    forecast_data['is_sunday'] = (forecast_data['dayofweek'] == 6).astype(int)
    
    # Month indicators for forecast data
    forecast_data['is_month_start'] = (forecast_data['dayofmonth'] <= 5).astype(int)
    forecast_data['is_month_end'] = (forecast_data['dayofmonth'] >= 25).astype(int)
    
    # Weather features for forecast data
    forecast_data['temp_range'] = forecast_data['tempmax'] - forecast_data['tempmin']
    forecast_data['comfortable_weather'] = ((forecast_data['temp'] >= 60) & (forecast_data['temp'] <= 75) & (forecast_data['precip'] <= 0.1)).astype(int)
    forecast_data['bad_weather'] = ((forecast_data['temp'] <= 40) | (forecast_data['temp'] >= 85) | (forecast_data['precip'] >= 0.5)).astype(int)
    
    # For lag features in forecast, use recent historical values
    recent_sales = train_data['Total_Sales'].tail(30).values
    
    # Add lag-like features to forecast using recent historical patterns
    if len(recent_sales) >= 14:
        forecast_data['sales_lag_1'] = recent_sales[-1]
        forecast_data['sales_lag_7'] = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales[-1]
        forecast_data['sales_lag_14'] = recent_sales[-14] if len(recent_sales) >= 14 else recent_sales[-1]
        
        # Rolling averages using recent data
        forecast_data['sales_rolling_7'] = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else np.mean(recent_sales)
        forecast_data['sales_rolling_14'] = np.mean(recent_sales[-14:]) if len(recent_sales) >= 14 else np.mean(recent_sales)
        forecast_data['sales_rolling_30'] = np.mean(recent_sales) if len(recent_sales) >= 30 else np.mean(recent_sales)
    else:
        # Fallback for limited data
        avg_sales = np.mean(recent_sales) if len(recent_sales) > 0 else 5000
        forecast_data['sales_lag_1'] = avg_sales
        forecast_data['sales_lag_7'] = avg_sales
        forecast_data['sales_lag_14'] = avg_sales
        forecast_data['sales_rolling_7'] = avg_sales
        forecast_data['sales_rolling_14'] = avg_sales
        forecast_data['sales_rolling_30'] = avg_sales
    
    # Fill NaN values in training data
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())
    
    # Fill NaN values in forecast data  
    numeric_cols_forecast = forecast_data.select_dtypes(include=[np.number]).columns
    forecast_data[numeric_cols_forecast] = forecast_data[numeric_cols_forecast].fillna(forecast_data[numeric_cols_forecast].median())
    
    return train_data, forecast_data

# Optimized model training with intelligent caching and reduced CV overhead
def train_models(train_data, features, target, company_id, model_params=None, cv_folds=3, test_size=0.2, force_retrain=False):
    """Train forecasting models with optimized performance"""
    
    # Quick check for existing models first
    if not force_retrain:
        existing_models = find_existing_models(company_id, features)
        
        if existing_models:
            # Create simplified model selection
            model_options = []
            for model_info in existing_models:
                timestamp = pd.to_datetime(model_info['timestamp']).strftime('%Y-%m-%d %H:%M')
                r2_score_val = model_info['metrics'].get('r2', 0)
                mae_score = model_info['metrics'].get('mae', 0)
                model_display = f"{model_info['model_name']} | {timestamp} | R²: {r2_score_val:.3f} | MAE: ${mae_score:,.0f}"
                model_options.append(model_display)
            
            with st.expander("📂 Use Existing Model?", expanded=False):
                use_existing = st.selectbox(
                    "Found existing models. Select one or train new:",
                    ["🆕 Train New Models"] + model_options[:3],  # Limit to top 3
                    help="Select existing model to save time"
                )
                
                if use_existing != "🆕 Train New Models":
                    selected_index = model_options.index(use_existing)
                    selected_model_info = existing_models[selected_index]
                    
                    # Convert to expected format
                    results = {}
                    model_data = selected_model_info['model_data']
                    results[model_data['model_name']] = {
                        'model': model_data['model'],
                        **model_data['metrics']
                    }
                    
                    st.success(f"✅ Using existing model: {selected_model_info['model_name']}")
                    return results
    
    # Data preprocessing
    if 'Is_Closed' in train_data.columns:
        train_data = train_data[train_data['Is_Closed'] == 0].copy()
    
    # Ensure we have the required features
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        st.error(f"❌ Missing features in training data: {missing_features}")
        return {}
    
    X = train_data[features].copy()
    y = train_data[target].copy()
    
    # Handle missing values efficiently
    if X.isnull().sum().sum() > 0:
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                if X[col].isnull().any():
                    mean_val = X[col].mean()
                    X[col] = X[col].fillna(mean_val if not pd.isna(mean_val) else 0)
            else:
                X[col] = X[col].fillna(0)
    
    # Smart feature selection for individual companies with limited data
    if len(X) < 100:  # Individual company
        # Remove features with too many zeros or low variance
        low_variance_features = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Remove features where >80% of values are the same
                most_common_ratio = (X[col] == X[col].mode().iloc[0]).sum() / len(X) if len(X[col].mode()) > 0 else 0
                if most_common_ratio > 0.8:
                    low_variance_features.append(col)
        
        # Keep essential features even if low variance
        essential_features = ['dayofweek', 'month', 'Is_Weekend', 'temp', 'Total_Sales']
        features_to_remove = [f for f in low_variance_features if f not in essential_features]
        
        if features_to_remove:
            X = X.drop(columns=features_to_remove)
            features = [f for f in features if f not in features_to_remove]
            if len(features_to_remove) <= 3:  # Only show if not too many
                # st.info(f"ℹ️ Removed {len(features_to_remove)} low-variance features for better individual company modeling.")
                pass
    
    # Remove rows with missing target values
    if y.isnull().any():
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
    
    # Check data sufficiency
    if len(X) < 10:
        st.error(f"❌ Insufficient data for training: only {len(X)} samples available")
        return {}
    
    # Adjust parameters for small datasets
    if len(X) < 50:
        # st.warning(f"⚠️ Limited training data: {len(X)} samples")
        test_size = min(0.2, max(0.1, 5/len(X)))
        cv_folds = min(cv_folds, max(2, len(X)//5))  # Reduce CV folds
    
    # Adjust parameters for individual companies with smaller datasets
    if len(X) < 100:  # Small dataset - individual company
        if model_params is None:
            model_params = {
                'XGBoost': {
                    'n_estimators': 80,  # Fewer estimators for small data
                    'random_state': 42, 
                    'learning_rate': 0.15,  # Higher learning rate for small data
                    'max_depth': 3,  # Shallower for small data to prevent overfitting
                    'subsample': 0.8,
                    'min_child_weight': 1,  # Lower for small data flexibility
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.01
                },
                'Gradient Boosting': {
                    'n_estimators': 80,  # Fewer estimators for small data
                    'random_state': 42, 
                    'learning_rate': 0.15,  # Higher learning rate for small data
                    'max_depth': 3,  # Shallower for small data
                    'subsample': 0.8,
                    'min_samples_leaf': 1,  # More flexible for small data
                    'min_samples_split': 2,  # More flexible for small data
                    'max_features': 'sqrt'
                }
            }
    else:  # Large dataset - combined locations
        if model_params is None:
            model_params = {
                'XGBoost': {
                    'n_estimators': 120,  # More estimators for large data
                    'random_state': 42, 
                    'learning_rate': 0.12,  # Standard learning rate
                    'max_depth': 5,  # Deeper for complex patterns
                    'subsample': 0.85,
                    'min_child_weight': 2,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.01
                },
                'Gradient Boosting': {
                    'n_estimators': 120,  # More estimators for large data
                    'random_state': 42, 
                    'learning_rate': 0.12,  # Standard learning rate
                    'max_depth': 5,  # Deeper for complex patterns
                    'subsample': 0.85,
                    'min_samples_leaf': 2,
                    'min_samples_split': 4,
                    'max_features': 'sqrt'
                }
            }
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        st.error(f"❌ Error splitting data: {str(e)}")
        return {}
    
    # Initialize models
    models = {
        'XGBoost': XGBRegressor(**model_params.get('XGBoost', {'n_estimators': 50, 'random_state': 42})),
        'Gradient Boosting': GradientBoostingRegressor(**model_params.get('Gradient Boosting', {'n_estimators': 50, 'random_state': 42}))
    }
    
    # Train and evaluate models with progress indicator
    results = {}
    features_hash = get_features_hash(features)
    
    with st.spinner("🤖 Training models..."):
        progress_container = st.container()
        
        for i, (name, model) in enumerate(models.items()):
            try:
                # Fit the model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate core metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Calculate MAPE safely
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else 0.0
                
                # Simplified cross-validation (only if dataset is large enough)
                if len(X) > 30:
                    try:
                        cv_scores_r2 = cross_val_score(model, X, y, cv=min(cv_folds, 3), scoring='r2')
                        cv_r2 = np.mean(cv_scores_r2)
                        cv_r2_std = np.std(cv_scores_r2)
                    except Exception:
                        cv_r2 = r2
                        cv_r2_std = 0
                else:
                    cv_r2 = r2
                    cv_r2_std = 0
                
                # Store metrics
                metrics = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mape': float(mape),
                    'cv_r2': float(cv_r2),
                    'cv_r2_std': float(cv_r2_std),
                    'train_score': float(model.score(X_train, y_train)),
                    'test_score': float(model.score(X_test, y_test)),
                    'training_time': float(training_time)
                }
                
                # Store results
                results[name] = {
                    'model': model,
                    **metrics
                }
                
                # Save model to disk
                filename = generate_model_filename(company_id, name, features_hash)
                model_path = save_model_with_metadata(
                    model, name, company_id, features, metrics, filename
                )
                
                if model_path:
                    results[name]['saved_path'] = model_path
                    results[name]['filename'] = filename
                
                with progress_container:
                    st.success(f"✅ {name}: R²={r2:.3f}, MAE=${mae:,.0f} (trained in {training_time:.1f}s)")
                
            except Exception as e:
                st.error(f"❌ Error training {name} model: {str(e)}")
                continue
    
    # Clear caches to reflect new models
    get_model_storage_info.clear()
    
    return results

# Optimized forecast generation
@st.cache_data
def generate_forecast(_model_results, historical_data, forecast_data, features, selected_model):
    """Generate forecast with confidence intervals and complete date coverage"""
    
    # Create copies
    historical_data_copy = historical_data.copy()
    forecast_data_copy = forecast_data.copy()
    
    # Ensure complete forecast date range (fill any missing dates)
    forecast_data_copy = ensure_complete_forecast_dates(forecast_data_copy)
    
    # Filter open days for historical data
    if 'Is_Closed' in historical_data_copy.columns:
        historical_open = historical_data_copy[historical_data_copy['Is_Closed'] == 0]
    else:
        historical_open = historical_data_copy.copy()
    
    # Separate open and closed days for forecast
    forecast_open = forecast_data_copy[forecast_data_copy['Is_Closed'] == 0] if 'Is_Closed' in forecast_data_copy.columns else forecast_data_copy.copy()
    forecast_closed = forecast_data_copy[forecast_data_copy['Is_Closed'] == 1] if 'Is_Closed' in forecast_data_copy.columns else pd.DataFrame()
    
    # Handle edge case where all forecast days might be closed
    if len(forecast_open) == 0:
        # If all days are marked as closed, still generate predictions for all days
        # This allows the model to predict what sales would be if the location were open
        forecast_open = forecast_data_copy.copy()
        forecast_closed = pd.DataFrame()  # Empty closed days since we're predicting for all
        
        # Add a note about this situation
        if 'Is_Closed' in forecast_data_copy.columns and forecast_data_copy['Is_Closed'].sum() == len(forecast_data_copy):
            pass  # All days marked as closed, but we'll still generate predictions
    
    # Prepare features for forecast data (only open days)
    X_forecast = forecast_open[features].fillna(forecast_open[features].mean())
    
    # Ensure we have at least some data to predict on
    if len(X_forecast) == 0:
        # Create a single row with default values as fallback
        default_row = {}
        for feature in features:
            if feature in ['dayofweek', 'dayofmonth', 'week', 'month', 'quarter']:
                default_row[feature] = 1  # Default date features
            elif feature == 'Is_Weekend':
                default_row[feature] = 0  # Weekday
            else:
                default_row[feature] = forecast_data_copy[feature].mean() if feature in forecast_data_copy.columns and not forecast_data_copy[feature].isna().all() else 0
        
        X_forecast = pd.DataFrame([default_row])
        
        # Create corresponding forecast display data
        forecast_open = pd.DataFrame({
            'Operational Date': [forecast_data_copy['Operational Date'].iloc[0] if len(forecast_data_copy) > 0 else pd.Timestamp('2025-05-15')],
            **default_row
        })
    
    # Get the selected model
    model = _model_results[selected_model]['model']
    
    # Make predictions only for open days
    forecasted_sales = model.predict(X_forecast)
    
    # Remove all artificial bounds - let the model predict naturally
    # Only ensure non-negative values
    forecasted_sales = np.maximum(forecasted_sales, 0)
    
    # Scale forecasts to match historical levels if they're too low - ENHANCED FOR INDIVIDUAL COMPANIES
    if len(historical_open) > 0:
        historical_mean = historical_open['Total_Sales'].mean()
        historical_median = historical_open['Total_Sales'].median()
        historical_std = historical_open['Total_Sales'].std()
        forecast_mean = forecasted_sales.mean()
        
        # For better scaling, use recent historical data if available
        if len(historical_open) > 30:
            recent_historical = historical_open.tail(30)  # Last 30 days
            recent_mean = recent_historical['Total_Sales'].mean()
            recent_median = recent_historical['Total_Sales'].median()
            # Weight recent data more heavily
            reference_value = (recent_mean * 0.7 + historical_mean * 0.3 + recent_median * 0.2 + historical_median * 0.1) / 1.3
        else:
            # For small datasets, use all available data
            reference_value = (historical_mean + historical_median) / 2
        
        # More intelligent scaling based on data characteristics
        if forecast_mean > 0:
            # Calculate how far off the forecast is
            ratio = forecast_mean / reference_value
            
            # Apply scaling if forecast is significantly different from historical
            if ratio < 0.7:  # Forecast too low
                scale_factor = (reference_value / forecast_mean) * 0.9  # Scale to 90% of reference
                forecasted_sales = forecasted_sales * scale_factor
                
            elif ratio > 1.4:  # Forecast too high
                scale_factor = (reference_value / forecast_mean) * 1.1  # Scale to 110% of reference
                forecasted_sales = forecasted_sales * scale_factor
        
        # For individual companies with very small data, ensure minimum reasonable levels
        if len(historical_open) < 50:
            min_reasonable = reference_value * 0.6  # At least 60% of historical average
            forecasted_sales = np.maximum(forecasted_sales, min_reasonable)
            
        # Ensure forecasts are within reasonable bounds of historical data
        historical_q25 = historical_open['Total_Sales'].quantile(0.25)
        historical_q75 = historical_open['Total_Sales'].quantile(0.75)
        
        # Cap extremely low values
        forecasted_sales = np.maximum(forecasted_sales, historical_q25 * 0.3)
        
        # Cap extremely high values (but allow for some growth)
        max_reasonable = historical_q75 * 2.0  # Allow up to 2x the 75th percentile
        forecasted_sales = np.minimum(forecasted_sales, max_reasonable)
    
    # Add realistic daily variability based on historical patterns - ENHANCED FOR INDIVIDUAL COMPANIES
    if len(historical_open) > 0 and len(forecasted_sales) > 1:
        historical_sales = historical_open['Total_Sales']
        
        # Calculate overall volatility from historical data
        historical_cv = historical_sales.std() / historical_sales.mean() if historical_sales.mean() > 0 else 0.3
        
        # Calculate day-of-week patterns from historical data
        dow_patterns = {}
        for dow in range(7):
            dow_sales = historical_sales[historical_open['Operational Date'].dt.dayofweek == dow]
            if len(dow_sales) > 0:
                dow_patterns[dow] = {
                    'mean': dow_sales.mean(),
                    'std': dow_sales.std(),
                    'variation': dow_sales.std() / dow_sales.mean() if dow_sales.mean() > 0 else historical_cv
                }
            else:
                # For individual companies with missing days, use overall pattern
                dow_patterns[dow] = {
                    'mean': historical_sales.mean(), 
                    'std': historical_sales.std(), 
                    'variation': historical_cv
                }
        
        # Apply day-of-week specific variations to forecasts
        for i, date in enumerate(forecast_open['Operational Date']):
            dow = date.dayofweek
            base_forecast = forecasted_sales[i]
            
            # Get day-specific variation pattern
            variation_factor = dow_patterns[dow]['variation']
            
            # Use historical coefficient of variation as baseline
            if len(historical_open) < 100:  # Individual company
                # Use actual historical variation but cap it reasonably
                variation_factor = min(variation_factor, historical_cv * 1.2, 0.5)  # Cap at 50%
            else:  # Combined company
                # Allow more variation for combined data
                variation_factor = min(variation_factor, historical_cv * 1.5, 0.7)  # Cap at 70%
            
            # Create realistic variation using multiple components
            np.random.seed(42 + i)  # Consistent randomness
            
            # 1. Weekly cycle (some days naturally higher/lower)
            weekly_cycle = 1.0 + 0.2 * np.sin(2 * np.pi * dow / 7 + 1.5)  # Phase shift for realism
            
            # 2. Trend component (slight ups and downs over time)
            trend_component = 1.0 + 0.1 * np.sin(2 * np.pi * i / len(forecasted_sales) * 3)  # 3 cycles over forecast period
            
            # 3. Random daily variation based on historical patterns
            daily_random = np.random.normal(1.0, variation_factor * 0.6)  # 60% of historical variation
            
            # 4. Business day effects (Monday/Friday often different)
            business_day_effect = 1.0
            if dow == 0:  # Monday - often lower
                business_day_effect = 0.9
            elif dow == 4:  # Friday - often higher
                business_day_effect = 1.1
            elif dow >= 5:  # Weekend
                business_day_effect = 0.85 if dow == 5 else 0.7  # Saturday > Sunday
            
            # Combine all factors
            total_variation = weekly_cycle * trend_component * daily_random * business_day_effect
            
            # Apply reasonable bounds based on historical data
            min_multiplier = max(0.4, 1.0 - historical_cv * 2)  # Don't go below 40% or 2 std devs
            max_multiplier = min(2.0, 1.0 + historical_cv * 2)  # Don't go above 200% or 2 std devs
            
            total_variation = np.clip(total_variation, min_multiplier, max_multiplier)
            
            forecasted_sales[i] = base_forecast * total_variation
    
    # Prepare historical data for display (no predictions needed)
    historical_display = historical_open.copy()
    historical_display['Actual_Sales'] = historical_display['Total_Sales']
    
    # Prepare forecast data for display - include both open and closed days
    forecast_display_open = forecast_open.copy()
    forecast_display_open['Forecasted_Sales'] = forecasted_sales
    
    # Add closed days with zero forecast
    if len(forecast_closed) > 0:
        forecast_display_closed = forecast_closed.copy()
        forecast_display_closed['Forecasted_Sales'] = 0  # Zero sales for closed days (red dots)
        
        # Combine open and closed forecast days
        forecast_display = pd.concat([forecast_display_open, forecast_display_closed]).sort_values('Operational Date').reset_index(drop=True)
        
        # Boost sales on days adjacent to closed days (realistic business pattern)
        for i in range(len(forecast_display)):
            if forecast_display.iloc[i]['Forecasted_Sales'] > 0:  # Open day
                # Check if previous or next day is closed
                has_adjacent_closure = False
                closure_boost = 1.0
                
                if i > 0 and forecast_display.iloc[i-1]['Forecasted_Sales'] == 0:
                    has_adjacent_closure = True
                    closure_boost *= 1.15  # 15% boost for day after closure
                if i < len(forecast_display)-1 and forecast_display.iloc[i+1]['Forecasted_Sales'] == 0:
                    has_adjacent_closure = True
                    closure_boost *= 1.1   # 10% boost for day before closure
                
                # Additional patterns for realistic business flow
                current_dow = forecast_display.iloc[i]['Operational Date'].dayofweek
                
                # Friday before weekend closure gets extra boost
                if current_dow == 4:  # Friday
                    weekend_closed = False
                    if i < len(forecast_display)-1 and forecast_display.iloc[i+1]['Forecasted_Sales'] == 0:  # Saturday closed
                        weekend_closed = True
                    if i < len(forecast_display)-2 and forecast_display.iloc[i+2]['Forecasted_Sales'] == 0:  # Sunday closed
                        weekend_closed = True
                    if weekend_closed:
                        closure_boost *= 1.2  # 20% Friday boost
                
                # Monday after weekend closure gets moderate boost
                elif current_dow == 0:  # Monday
                    weekend_closed = False
                    if i > 0 and forecast_display.iloc[i-1]['Forecasted_Sales'] == 0:  # Sunday closed
                        weekend_closed = True
                    if i > 1 and forecast_display.iloc[i-2]['Forecasted_Sales'] == 0:  # Saturday closed
                        weekend_closed = True
                    if weekend_closed:
                        closure_boost *= 1.1  # 10% Monday recovery boost
                
                # Apply the boost but cap it reasonably
                if closure_boost > 1.0:
                    final_boost = min(closure_boost, 1.4)  # Cap total boost at 40%
                    forecast_display.loc[i, 'Forecasted_Sales'] *= final_boost
    else:
        forecast_display = forecast_display_open
    
    # **ENSURE ALL CLOSED DAYS SHOW ZERO SALES (RED DOTS)**
    # Mark any days marked as closed with exactly zero sales for proper red dot display
    if 'Is_Closed' in forecast_display.columns:
        forecast_display.loc[forecast_display['Is_Closed'] == 1, 'Forecasted_Sales'] = 0
    
    # **NEW: Include ALL historical data (including closed days) for complete timeline**
    historical_display_complete = historical_data_copy.copy()
    historical_display_complete['Actual_Sales'] = historical_display_complete['Total_Sales']
    
    # Create weekly aggregations efficiently
    historical_weekly = historical_display_complete.groupby(
        historical_display_complete['Operational Date'].dt.isocalendar().week
    )['Actual_Sales'].sum().reset_index()
    historical_weekly.columns = ['Week', 'Actual_Sales']
    
    forecast_weekly = forecast_display.groupby(
        forecast_display['Operational Date'].dt.isocalendar().week
    )['Forecasted_Sales'].sum().reset_index()
    forecast_weekly.columns = ['Week', 'Forecasted_Sales']
    
    # **FINAL VALIDATION FOR INDIVIDUAL COMPANIES**
    # Ensure forecasts are reasonable for individual companies
    if len(historical_open) > 0:
        # Analyze recent trends to make forecasts more realistic
        if len(historical_open) > 14:
            # Get recent trend (last 2 weeks vs previous 2 weeks)
            recent_period = historical_open.tail(14)['Total_Sales'].mean()
            previous_period = historical_open.iloc[-28:-14]['Total_Sales'].mean() if len(historical_open) > 28 else historical_open['Total_Sales'].mean()
            
            if previous_period > 0:
                trend_factor = recent_period / previous_period
                # Apply trend but cap it (don't want extreme trend extrapolation)
                trend_factor = np.clip(trend_factor, 0.8, 1.3)  # Max 30% up/down trend
                forecasted_sales = forecasted_sales * trend_factor
        
        # Final range validation
        historical_range = historical_open['Total_Sales'].quantile([0.1, 0.9])
        historical_q10, historical_q90 = historical_range.iloc[0], historical_range.iloc[1]
        
        # Check if forecasts are completely outside reasonable range
        forecast_median = np.median(forecasted_sales[forecasted_sales > 0])
        
        if forecast_median < historical_q10 * 0.5:  # Forecasts way too low
            boost_factor = (historical_q10 * 0.8) / forecast_median if forecast_median > 0 else 2.0
            forecasted_sales = forecasted_sales * min(boost_factor, 2.5)  # Cap boost at 2.5x
            
        elif forecast_median > historical_q90 * 1.8:  # Forecasts way too high
            reduction_factor = (historical_q90 * 1.4) / forecast_median
            forecasted_sales = forecasted_sales * max(reduction_factor, 0.6)  # Cap reduction at 40%
    
    # Ensure minimum forecast for business viability (individual companies need some sales)
    if len(forecasted_sales) > 0:
        # More intelligent minimum based on historical data
        if len(historical_open) > 0:
            historical_min_open = historical_open[historical_open['Total_Sales'] > 0]['Total_Sales'].min()
            min_viable_sales = max(100, historical_min_open * 0.3)  # At least 30% of historical minimum
        else:
            min_viable_sales = 100 if len(historical_open) < 50 else 50
        forecasted_sales = np.maximum(forecasted_sales, min_viable_sales)
    
    return historical_display_complete, forecast_display, historical_weekly, forecast_weekly

def ensure_complete_forecast_dates(forecast_data):
    """Ensure forecast data covers all dates in the expected range and mark missing dates as closed"""
    try:
        if len(forecast_data) == 0:
            return forecast_data
        
        # Get the full date range for forecast
        min_date = forecast_data['Operational Date'].min()
        max_date = forecast_data['Operational Date'].max()
        
        # Create complete date range
        complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create a DataFrame with all dates
        complete_df = pd.DataFrame({'Operational Date': complete_date_range})
        
        # Merge with existing data, filling missing values
        forecast_data = pd.merge(complete_df, forecast_data, on='Operational Date', how='left')
        
        # Fill missing forecast data appropriately
        # Mark missing days as closed (zero forecasted sales)
        if 'Is_Closed' not in forecast_data.columns:
            forecast_data['Is_Closed'] = 0
        
        # **NEW**: Mark missing dates as closed days
        missing_dates_mask = forecast_data['tempmax'].isna()  # Use weather data to identify missing dates
        forecast_data.loc[missing_dates_mask, 'Is_Closed'] = 1
        
        # Fill date features for missing days
        forecast_data['dayofweek'] = forecast_data['Operational Date'].dt.dayofweek
        forecast_data['dayofmonth'] = forecast_data['Operational Date'].dt.day
        forecast_data['week'] = forecast_data['Operational Date'].dt.isocalendar().week
        forecast_data['month'] = forecast_data['Operational Date'].dt.month
        forecast_data['quarter'] = forecast_data['Operational Date'].dt.quarter
        forecast_data['Is_Weekend'] = (forecast_data['dayofweek'] >= 5).astype(int)
        
        # Fill weather data with interpolation for missing dates
        weather_columns = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'cloudcover', 'solarradiation', 'uvindex']
        for col in weather_columns:
            if col in forecast_data.columns:
                # Interpolate missing weather values
                forecast_data[col] = forecast_data[col].interpolate(method='linear', limit_direction='both')
                forecast_data[col] = forecast_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use reasonable defaults
                if forecast_data[col].isna().any():
                    weather_defaults = {
                        'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
                        'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
                        'solarradiation': 200, 'uvindex': 6
                    }
                    forecast_data[col] = forecast_data[col].fillna(weather_defaults.get(col, 0))
        
        # Apply intelligent closure prediction for remaining missing days
        # For weekends, higher probability of closure (especially Sundays in some businesses)
        forecast_data.loc[(forecast_data['Is_Closed'].isna()) & (forecast_data['dayofweek'] == 0), 'Is_Closed'] = 1  # Sunday - often closed
        forecast_data.loc[(forecast_data['Is_Closed'].isna()) & (forecast_data['dayofweek'] == 6), 'Is_Closed'] = 0  # Saturday - usually open
        
        # Fill remaining with 0 (assume open unless specified)
        forecast_data['Is_Closed'] = forecast_data['Is_Closed'].fillna(0)
        
        # Sort by date
        forecast_data = forecast_data.sort_values('Operational Date').reset_index(drop=True)
        
        return forecast_data
        
    except Exception as e:
        st.error(f"❌ Error ensuring complete forecast dates: {str(e)}")
        return forecast_data

# Optimized model deletion
def delete_models(company_id=None, model_filenames=None):
    """Delete models with optimized file operations"""
    try:
        deleted_count = 0
        deleted_files = []
        
        model_files = list_model_files()
        
        if model_filenames:
            # Delete specific models
            for filename in model_filenames:
                if filename in model_files:
                    model_path = get_model_path(filename)
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        deleted_count += 1
                        deleted_files.append(filename)
        
        elif company_id:
            # Delete all models for a company
            for filename in model_files:
                if filename.startswith(f"model_{company_id}_"):
                    model_path = get_model_path(filename)
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        deleted_count += 1
                        deleted_files.append(filename)
        
        else:
            # Delete all models
            for filename in model_files:
                model_path = get_model_path(filename)
                if os.path.exists(model_path):
                    os.remove(model_path)
                    deleted_count += 1
                    deleted_files.append(filename)
        
        # Clear caches
        list_model_files.clear()
        get_model_storage_info.clear()
        
        return deleted_count, deleted_files
        
    except Exception as e:
        st.error(f"❌ Error deleting models: {str(e)}")
        return 0, []

# Optimized model info retrieval
@st.cache_data(ttl=600)
def get_all_saved_models():
    """Get information about all saved models - cached and optimized"""
    try:
        model_files = list_model_files()
        if not model_files:
            return []
            
        all_models = []
        
        for filename in model_files:
            model_data = load_model_with_metadata(filename)
            if model_data:
                file_size = get_file_size(get_model_path(filename)) / (1024 * 1024)  # MB
                
                model_info = {
                    'filename': filename,
                    'company_id': model_data.get('company_id', 'Unknown'),
                    'model_name': model_data.get('model_name', 'Unknown'),
                    'timestamp': model_data.get('timestamp', 'Unknown'),
                    'r2_score': model_data.get('metrics', {}).get('r2', 0),
                    'mae': model_data.get('metrics', {}).get('mae', 0),
                    'file_size_mb': file_size
                }
                all_models.append(model_info)
        
        # Sort by timestamp (newest first)
        all_models.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_models
        
    except Exception as e:
        st.error(f"Error getting model information: {str(e)}")
        return []

# Optimized main function with better performance
def main():
    # Clean Header - more compact
    st.markdown("""
    <div class='clean-header'>
        <div class='clean-header-content'>
            <div class='clean-header-icon'>📊</div>
            <div>
                <h1 class='clean-header-title'>Sales Forecasting</h1>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # More compact layout with much smaller left panel
    col_config, col_forecast = st.columns([1, 5], gap="small")  # Changed from [1, 4] to [1, 5] for much smaller config panel
    
    with col_config:
        # Location Selection - compact section
        st.markdown("""
        <div class="compact-section">
            <h3>📍 Location</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_company = st.selectbox(
            "Choose",
            list(COMPANY_INFO.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        # Load data with progress indicator
        with st.spinner("Loading..."):
            try:
                sales_data, weather_data, company_config = load_data(selected_company)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
        
        # Compact info display
        try:
            st.markdown(f"""
            <div class="info-box-compact">
                <strong>{company_config['name']}</strong><br>
                📊 {len(sales_data):,} records
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
        
        # Forecast Settings - compact section
        st.markdown("""
        <div class="compact-section">
            <h3>⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Engineer features with caching
            train_data, forecast_data = engineer_features(sales_data, weather_data, company_config)
            
            # Date range selection - more compact
            try:
                forecast_dates = forecast_data['Operational Date'].dropna()
                if len(forecast_dates) > 0:
                    forecast_start = forecast_dates.min()
                    forecast_end = forecast_dates.max()
                else:
                    forecast_start = pd.Timestamp('2025-05-15')
                    forecast_end = pd.Timestamp('2025-05-27')
                
                if pd.isna(forecast_start):
                    forecast_start = pd.Timestamp('2025-05-15')
                if pd.isna(forecast_end):
                    forecast_end = pd.Timestamp('2025-05-27')
                    
            except Exception as e:
                st.error(f"❌ Error processing forecast dates: {str(e)}")
                forecast_start = pd.Timestamp('2025-05-15')
                forecast_end = pd.Timestamp('2025-05-27')
            
            # Compact date inputs
            col_dates = st.columns(2)
            with col_dates[0]:
                start_date = st.date_input("Start", value=forecast_start.date(), min_value=forecast_start.date(), max_value=forecast_end.date())
            with col_dates[1]:
                end_date = st.date_input("End", value=forecast_end.date(), min_value=start_date, max_value=forecast_end.date())
            
            # Convert to datetime safely
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
            except Exception as e:
                st.error(f"❌ Error converting dates: {str(e)}")
                start_date = pd.Timestamp('2025-05-15')
                end_date = pd.Timestamp('2025-05-27')
            
            # Model selection
            model_option = st.selectbox("Model", ["XGBoost", "Gradient Boosting"], index=1)
            
            # Feature information
            available_features = ['dayofweek', 'dayofmonth', 'week', 'month', 'quarter', 'Is_Weekend',
                                'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                                'precipprob', 'cloudcover', 'solarradiation', 'uvindex',
                                'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 
                                'dow_avg_sales', 'sales_7d_trend', 'sales_30d_trend',
                                'is_start_of_month', 'is_end_of_month', 'is_mid_month',
                                'is_monday', 'is_friday', 'day_sin', 'day_cos', 
                                'month_sin', 'month_cos']
            
            selected_features = [feature for feature in available_features 
                               if feature in train_data.columns or feature in ['dayofweek', 'dayofmonth', 'week', 'month', 'quarter', 'Is_Weekend']]
            
            # Display options - compact
            col_opts = st.columns(2)
            with col_opts[0]:
                include_historical = st.checkbox("Historical", value=True)
            with col_opts[1]:
                confidence_interval = st.checkbox("Confidence", value=False)
        
        except Exception as e:
            st.error(f"❌ Error in forecast settings: {str(e)}")
            st.stop()
        
        # Chart Controls - compact section
        st.markdown("""
        <div class="compact-section">
            <h3>📊 Chart</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick time period selection
        time_period = st.selectbox(
            "View",
            ["Last 1 Month + Forecast", "Last 2 Weeks + Forecast", "Last 1 Week + Forecast", "Forecast Only", "All Data"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Chart customization options
        col_chart = st.columns(2)
        with col_chart[0]:
            show_markers = st.checkbox("Markers", value=True)
        with col_chart[1]:
            show_grid = st.checkbox("Grid", value=True)
        
        # Advanced chart options in compact expander
        with st.expander("🎛️ Advanced", expanded=False):
            chart_height = st.slider("Height", 300, 600, 400, 50)
            line_width = st.slider("Line Width", 1, 5, 2)
            
            col_colors = st.columns(2)
            with col_colors[0]:
                historical_color = st.color_picker("Historical", "#1f77b4")
            with col_colors[1]:
                forecast_color = st.color_picker("Forecast", "#ff7f0e")
        
        # Model Management - compact section
        st.markdown("""
        <div class="compact-section">
            <h3>🤖 Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get storage info
        storage_info = get_model_storage_info()
        st.markdown(f"""
        <div class="info-box-compact">
            <strong>Saved:</strong> {storage_info['total_models']} models ({storage_info['total_size_mb']:.1f} MB)
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced model management
        col_mgmt = st.columns(2)
        with col_mgmt[0]:
            if st.button("🗑️ Clear", key="clear_models"):
                deleted_count, _ = delete_models(company_id=company_config['id'])
                if deleted_count > 0:
                    st.success(f"✅ Deleted {deleted_count} model(s)")
                    if 'model_results' in st.session_state:
                        del st.session_state.model_results
                    st.rerun()
                else:
                    st.info("ℹ️ No models found")
        
        with col_mgmt[1]:
            # Load existing model button
            if st.button("📂 Load", key="load_models"):
                existing_models = find_existing_models(company_config['id'], selected_features)
                if existing_models:
                    # Use the most recent model
                    latest_model = existing_models[0]
                    st.session_state.model_results = {
                        latest_model['model_name']: {
                            'model': latest_model['model_data']['model'],
                            **latest_model['metrics']
                        }
                    }
                    st.success(f"✅ Loaded: {latest_model['model_name']}")
                    st.rerun()
                else:
                    st.info("ℹ️ No saved models found")
        
        # Generate Button - compact
        generate_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    with col_forecast:
        if generate_btn or 'model_results' in st.session_state:
            # Train models
            if 'model_results' not in st.session_state or generate_btn:
                try:
                    st.session_state.model_results = train_models(
                        train_data, 
                        selected_features, 
                        'Total_Sales',
                        company_config['id']
                    )
                except Exception as e:
                    st.error(f"❌ Error training models for {company_config['name']}: {str(e)}")
                    st.error("💡 **Suggestion**: Try using 'All Locations Combined' for more reliable forecasting, or check if data is available for this location.")
                    st.stop()
            
            # Check if any models were successfully trained
            if not st.session_state.model_results:
                st.error("❌ No models were successfully trained. Please check your data and try again.")
                st.info("💡 **Try**: Select 'All Locations Combined' for more robust forecasting with larger datasets.")
                st.stop()
            
            # Check if selected model exists
            if model_option not in st.session_state.model_results:
                available_models = list(st.session_state.model_results.keys())
                if available_models:
                    st.warning(f"⚠️ Selected model '{model_option}' not available. Using '{available_models[0]}' instead.")
                    model_option = available_models[0]
                else:
                    st.error("❌ No trained models available for forecasting.")
                    st.stop()
            
            # Filter forecast data by date range
            try:
                # Remove any NaT values before filtering
                valid_forecast_data = forecast_data.dropna(subset=['Operational Date'])
                forecast_data_filtered = valid_forecast_data[
                    (valid_forecast_data['Operational Date'] >= start_date) & 
                    (valid_forecast_data['Operational Date'] <= end_date)
                ]
                
                if len(forecast_data_filtered) == 0:
                    st.warning("⚠️ No forecast data available for the selected date range.")
                    # Use original forecast data as fallback
                    forecast_data_filtered = valid_forecast_data
                    
            except Exception as e:
                st.error(f"❌ Error filtering forecast data: {str(e)}")
                forecast_data_filtered = forecast_data.dropna(subset=['Operational Date'])
            
            # Generate forecast
            historical_all, forecast_all, historical_weekly, forecast_weekly = generate_forecast(
                st.session_state.model_results, train_data, forecast_data_filtered, 
                selected_features, model_option
            )
        
            # Display results - no section header wrapper
            st.markdown("### 📊 Forecast Results")
            
            # Validate forecast results for individual companies
            if len(forecast_all) == 0:
                st.error(f"❌ No forecast data generated for {company_config['name']}. Please check date range and data availability.")
                st.info("💡 **Try**: Adjust date range or select 'All Locations Combined' for more robust forecasting.")
                st.stop()
                
            if forecast_all['Forecasted_Sales'].isna().all():
                st.error(f"❌ All forecast values are invalid for {company_config['name']}. Data quality issues detected.")
                st.info("💡 **Try**: Select 'All Locations Combined' for more stable predictions.")
                st.stop()
            
            # Interactive features guide
            with st.expander("💡 Interactive Chart Guide", expanded=False):
                st.markdown("""
                **🎯 Quick Actions:**
                - **Scroll** to zoom in/out on the chart
                - **Double-click** to reset zoom to default view
                - **Drag** to pan around the chart
                - **Click & drag** to select a specific time range
                
                **📅 Time Controls:**
                - Use **date range buttons** (7D, 14D, 1M, 3M, 6M, All) for quick navigation
                - Use the **range slider** at the bottom to navigate through time
                - **Quick View** dropdown on the left for preset time periods
                
                **🎨 Customization:**
                - Adjust chart height, colors, and line styles in the sidebar
                - Toggle markers and grid lines for cleaner views
                - Export charts as PNG, HTML, or download data as CSV
                
                **📊 Data Points:**
                - **Blue markers**: Historical sales (red dots = closed days)
                - **Orange markers**: Forecasted sales (dark red dots = forecasted closed days)
                - **Gray dotted line**: Connection between historical and forecast data
                """)
            
            # Main forecast chart - no card wrapper
            if include_historical:
                fig, config = create_historical_forecast_chart(
                    historical_all, forecast_all,
                    f"Sales Forecast - {company_config['name']}",
                    historical_color=historical_color,
                    forecast_color=forecast_color,
                    show_markers=show_markers,
                    show_grid=show_grid,
                    chart_height=chart_height,
                    line_width=line_width,
                    time_period=time_period
                )
            else:
                fig, config = create_line_chart(
                    forecast_all, 'Operational Date', 'Forecasted_Sales',
                    f"Sales Forecast - {company_config['name']}", "Date", "Forecasted Sales ($)"
                )
            
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Handle export actions
            if 'export_png' in locals() and export_png:
                try:
                    import plotly.io as pio
                    img_bytes = pio.to_image(fig, format="png", width=1200, height=600, scale=2)
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes,
                        file_name=f"sales_forecast_{company_config['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"PNG export error: {str(e)}")
            
            if 'export_html' in locals() and export_html:
                try:
                    html_bytes = fig.to_html(include_plotlyjs=True).encode()
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_bytes,
                        file_name=f"sales_forecast_{company_config['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"HTML export error: {str(e)}")
            
            if 'export_data' in locals() and export_data:
                try:
                    # Combine historical and forecast data for export
                    export_df = pd.DataFrame()
                    
                    if len(historical_all) > 0:
                        hist_export = historical_all[['Operational Date', 'Actual_Sales']].copy()
                        hist_export['Data_Type'] = 'Historical'
                        hist_export['Value'] = hist_export['Actual_Sales']
                        hist_export = hist_export[['Operational Date', 'Value', 'Data_Type']]
                        export_df = pd.concat([export_df, hist_export], ignore_index=True)
                    
                    if len(forecast_all) > 0:
                        forecast_export = forecast_all[['Operational Date', 'Forecasted_Sales']].copy()
                        forecast_export['Data_Type'] = 'Forecast'
                        forecast_export['Value'] = forecast_export['Forecasted_Sales']
                        forecast_export = forecast_export[['Operational Date', 'Value', 'Data_Type']]
                        export_df = pd.concat([export_df, forecast_export], ignore_index=True)
                    
                    csv_bytes = export_df.to_csv(index=False).encode()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_bytes,
                        file_name=f"sales_data_{company_config['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"CSV export error: {str(e)}")
        
            # Summary metrics - more compact with historical comparison
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_forecast = forecast_all['Forecasted_Sales'].sum()
                st.markdown(f"""
                <div class="metric-card-compact">
                    <div class="metric-value-compact">${total_forecast/1000:.0f}K</div>
                    <div class="metric-label-compact">Total Forecast</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_daily = forecast_all['Forecasted_Sales'].mean()
                st.markdown(f"""
                <div class="metric-card-compact">
                    <div class="metric-value-compact">${avg_daily/1000:.1f}K</div>
                    <div class="metric-label-compact">Daily Avg</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Compare forecast vs historical average
                if len(historical_all) > 0:
                    historical_avg = historical_all['Total_Sales'].mean()
                    forecast_avg_open = forecast_all[forecast_all['Forecasted_Sales'] > 0]['Forecasted_Sales'].mean()
                    comparison_ratio = (forecast_avg_open / historical_avg) if historical_avg > 0 else 1
                    comparison_text = f"{comparison_ratio:.1f}x"
                    comparison_color = "green" if comparison_ratio >= 0.8 else "orange" if comparison_ratio >= 0.5 else "red"
                else:
                    comparison_text = "N/A"
                    comparison_color = "gray"
                
                st.markdown(f"""
                <div class="metric-card-compact">
                    <div class="metric-value-compact" style="color: {comparison_color};">{comparison_text}</div>
                    <div class="metric-label-compact">vs Historical</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                model_r2 = st.session_state.model_results[model_option]['r2']
                st.markdown(f"""
                <div class="metric-card-compact">
                    <div class="metric-value-compact">{model_r2:.2f}</div>
                    <div class="metric-label-compact">R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional forecast insights - REMOVED FOR CLEANER UI
            # Users don't need detailed technical analysis
            pass
        
            # Feature importance (optional) - compact
            if st.checkbox("Show Feature Importance", value=False):
                feature_importance = get_feature_importance(
                    st.session_state.model_results, selected_features, model_option
                )
                
                if len(feature_importance) > 0:
                    st.markdown("#### 🎯 Feature Importance")
                    
                    # Show top 5 features
                    top_features = feature_importance.head(5)
                    for idx, row in top_features.iterrows():
                        importance_pct = (row['Importance'] / feature_importance['Importance'].sum() * 100)
                        st.markdown(f"• **{row['Feature']}**: {importance_pct:.1f}% influence")

# Enhanced data validation functions
def validate_weather_data_usage(company_config, weather_data):
    """Validate that weather data is properly loaded and used"""
    company_id = company_config["id"]
    status = WEATHER_DATA_STATUS.get(company_id, "Unknown")
    
    if weather_data is None:
        return False, f"No weather data loaded for company {company_id}"
    
    # Check if weather data has required columns
    required_weather_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                            'precipprob', 'cloudcover', 'solarradiation', 'uvindex']
    missing_cols = [col for col in required_weather_cols if col not in weather_data.columns]
    
    if missing_cols:
        return False, f"Missing weather columns: {missing_cols}"
    
    # Check data quality
    if len(weather_data) == 0:
        return False, "Empty weather dataset"
    
    return True, status

def validate_sales_weather_merge(sales_data, weather_data):
    """Validate that sales and weather data are properly aligned - smart forecast detection"""
    if weather_data is None:
        return True, "Using default weather data"
    
    # Get date ranges
    sales_dates = set(sales_data['Operational Date'].dt.date)
    weather_dates = set(weather_data['Operational Date'].dt.date)
    
    # Determine if this is forecast weather data or historical weather data
    sales_min_date = min(sales_dates)
    sales_max_date = max(sales_dates)
    weather_min_date = min(weather_dates)
    weather_max_date = max(weather_dates)
    
    # Check if weather data is in the future (forecast data)
    is_forecast_weather = weather_min_date > sales_max_date
    
    if is_forecast_weather:
        # For forecast weather, we expect NO overlap with historical sales data
        overlap = len(sales_dates.intersection(weather_dates))
        if overlap == 0:
            days_ahead = (weather_min_date - sales_max_date).days
            return True, f"Forecast weather data: {len(weather_dates)} days starting {days_ahead} days ahead"
        else:
            return True, f"Mixed historical/forecast weather data: {overlap} overlapping days"
    
    else:
        # For historical weather, we expect significant overlap with sales data
        overlap = len(sales_dates.intersection(weather_dates))
        total_sales_dates = len(sales_dates)
        
        coverage = overlap / total_sales_dates if total_sales_dates > 0 else 0
        
        if coverage < 0.3:  # Lowered threshold for more lenient validation
            return False, f"Poor historical weather data coverage: {coverage:.1%} of sales dates"
        
        return True, f"Good historical weather coverage: {coverage:.1%} of dates"

def extend_kaapse_kaap_weather_forecast(historical_weather_data):
    """Extend Kaapse Kaap weather data for forecasting next 1-2 weeks using historical patterns"""
    try:
        if historical_weather_data is None or len(historical_weather_data) == 0:
            return create_default_forecast_weather()
        
        # Parse dates
        historical_weather_data['Operational Date'] = pd.to_datetime(historical_weather_data['Operational Date'])
        
        # Get the last date in historical data
        last_date = historical_weather_data['Operational Date'].max()
        
        # Create forecast dates for next 2 weeks (14 days)
        forecast_start = pd.Timestamp('2025-05-15')  # Standard forecast period
        forecast_end = pd.Timestamp('2025-05-27')   # Standard forecast period (13 days)
        
        # Generate forecast dates
        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
        
        # Calculate seasonal adjustments (winter historical -> spring forecast)
        # Historical data is from winter (Dec-Jan), forecast is for spring (May)
        seasonal_adjustments = {
            'tempmax': +15.0,  # Warmer in spring
            'tempmin': +8.0,   # Warmer nights
            'temp': +12.0,     # Overall warmer
            'humidity': -10.0, # Less humid in spring
            'precip': -0.1,    # Less precipitation
            'precipprob': -20, # Lower chance of rain
            'cloudcover': -15.0, # Less cloudy
            'solarradiation': +200.0, # More sunshine
            'uvindex': +4      # Higher UV in spring
        }
        
        # Calculate historical weather patterns
        weather_stats = {}
        for col in ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'cloudcover', 'solarradiation', 'uvindex']:
            if col in historical_weather_data.columns:
                weather_stats[col] = {
                    'mean': historical_weather_data[col].mean(),
                    'std': historical_weather_data[col].std(),
                    'min': historical_weather_data[col].min(),
                    'max': historical_weather_data[col].max()
                }
        
        # Generate forecast weather data
        forecast_weather = []
        
        for i, date in enumerate(forecast_dates):
            day_weather = {'Operational Date': date}
            
            for col in ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'cloudcover', 'solarradiation', 'uvindex']:
                if col in weather_stats:
                    # Base value from historical mean
                    base_value = weather_stats[col]['mean']
                    
                    # Apply seasonal adjustment
                    adjusted_value = base_value + seasonal_adjustments.get(col, 0)
                    
                    # Add some variation based on historical std (±20%)
                    variation = weather_stats[col]['std'] * 0.2 * (1 if i % 2 == 0 else -1)
                    final_value = adjusted_value + variation
                    
                    # Apply reasonable bounds
                    if col in ['tempmax', 'tempmin', 'temp']:
                        final_value = max(30, min(80, final_value))  # Reasonable temperature range
                    elif col == 'humidity':
                        final_value = max(30, min(95, final_value))  # Humidity range
                    elif col == 'precip':
                        final_value = max(0, min(2.0, final_value))  # Precipitation range
                    elif col == 'precipprob':
                        final_value = max(0, min(100, final_value))  # Probability range
                    elif col == 'cloudcover':
                        final_value = max(0, min(100, final_value))  # Cloud cover range
                    elif col == 'solarradiation':
                        final_value = max(50, min(400, final_value))  # Solar radiation range
                    elif col == 'uvindex':
                        final_value = max(0, min(11, final_value))  # UV index range
                    
                    day_weather[col] = round(final_value, 1)
                else:
                    # Default values if column not found
                    defaults = {
                        'tempmax': 65.0, 'tempmin': 45.0, 'temp': 55.0, 'humidity': 75.0,
                        'precip': 0.1, 'precipprob': 30, 'cloudcover': 60,
                        'solarradiation': 200, 'uvindex': 6
                    }
                    day_weather[col] = defaults.get(col, 0)
            
            forecast_weather.append(day_weather)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecast_weather)
        
        # Combine historical + forecast (for context) and return only forecast period
        return forecast_df
        
    except Exception as e:
        st.warning(f"⚠️ Could not extend Kaapse Kaap weather forecast: {str(e)}. Using default weather.")
        return create_default_forecast_weather()

def create_default_forecast_weather():
    """Create default forecast weather for May 15-27, 2025"""
    forecast_start = pd.Timestamp('2025-05-15')
    forecast_end = pd.Timestamp('2025-05-27')
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
    
    default_weather = []
    for i, date in enumerate(forecast_dates):
        # Create realistic spring weather patterns
        base_temp = 58 + (i % 7) * 2  # Temperature cycle
        day_weather = {
            'Operational Date': date,
            'tempmax': base_temp + 8 + (i % 3),
            'tempmin': base_temp - 12 + (i % 2),
            'temp': base_temp,
            'humidity': 70 + (i % 5) * 3,
            'precip': 0.1 if i % 4 == 0 else 0.0,  # Rain every 4th day
            'precipprob': 25 + (i % 3) * 15,
            'cloudcover': 50 + (i % 4) * 10,
            'solarradiation': 220 + (i % 6) * 20,
            'uvindex': 6 + (i % 3)
        }
        default_weather.append(day_weather)
    
    return pd.DataFrame(default_weather)

# Optimized chart creation with better performance
@st.cache_data
def create_historical_forecast_chart(historical_data, forecast_data, title="Sales Forecast", 
                                   historical_color="#1f77b4", forecast_color="#ff7f0e", 
                                   show_markers=True, show_grid=True, chart_height=400, line_width=2,
                                   time_period="Last 1 Month + Forecast"):
    """Create historical and forecast chart with complete date coverage and clear closed day indicators"""
    
    fig = go.Figure()
    
    # Add historical data if available
    if len(historical_data) > 0:
        # Create enhanced marker colors and sizes for better closed day visibility
        historical_colors = []
        historical_sizes = []
        historical_symbols = []
        hover_texts = []
        
        for sales in historical_data['Total_Sales']:
            if sales == 0:
                historical_colors.append('red')
                historical_sizes.append(12)  # Larger for closed days
                historical_symbols.append('circle')
                hover_texts.append('CLOSED DAY')
            else:
                historical_colors.append(historical_color)
                historical_sizes.append(8)  # Regular size for open days
                historical_symbols.append('circle')
                hover_texts.append('Open Day')
        
        # Add historical sales as a continuous line with enhanced closed day markers
        fig.add_trace(go.Scatter(
            x=historical_data['Operational Date'],
            y=historical_data['Total_Sales'],
            mode='lines+markers' if show_markers else 'lines',
            name='Historical Sales',
            line=dict(color=historical_color, width=line_width),
            marker=dict(
                size=historical_sizes,
                color=historical_colors,
                symbol=historical_symbols,
                line=dict(width=2, color='white'),  # White borders for visibility
                opacity=0.9
            ),
            hovertemplate='<b>%{x}</b><br>' + 
                         'Sales: $%{y:,.0f}' + 
                         '<br><b>%{text}</b><extra></extra>',
            text=hover_texts
        ))
    
    # Add forecast data
    if len(forecast_data) > 0:
        # Create enhanced marker colors and sizes for forecast closed days
        forecast_colors = []
        forecast_sizes = []
        forecast_symbols = []
        forecast_hover_texts = []
        
        for sales in forecast_data['Forecasted_Sales']:
            if sales == 0:
                forecast_colors.append('darkred')
                forecast_sizes.append(12)  # Larger for closed days
                forecast_symbols.append('circle')
                forecast_hover_texts.append('FORECASTED CLOSED DAY')
            else:
                forecast_colors.append(forecast_color)
                forecast_sizes.append(8)  # Regular size for open days
                forecast_symbols.append('circle')
                forecast_hover_texts.append('Forecasted Open Day')
        
        # Add forecasted sales as a continuous solid line with enhanced closed day markers
        fig.add_trace(go.Scatter(
            x=forecast_data['Operational Date'],
            y=forecast_data['Forecasted_Sales'],
            mode='lines+markers' if show_markers else 'lines',
            name='Forecasted Sales',
            line=dict(color=forecast_color, width=line_width),  # Solid line
            marker=dict(
                size=forecast_sizes,
                color=forecast_colors,
                symbol=forecast_symbols,
                line=dict(width=2, color='white'),  # White borders for visibility
                opacity=0.9
            ),
            hovertemplate='<b>%{x}</b><br>' + 
                         'Forecast: $%{y:,.0f}' + 
                         '<br><b>%{text}</b><extra></extra>',
            text=forecast_hover_texts
        ))
    
    # Add connection line between historical and forecast data if both exist
    if len(historical_data) > 0 and len(forecast_data) > 0:
        # Get the last historical point and first forecast point
        last_historical = historical_data.iloc[-1]
        first_forecast = forecast_data.iloc[0]
        
        # Add a connecting line
        fig.add_trace(go.Scatter(
            x=[last_historical['Operational Date'], first_forecast['Operational Date']],
            y=[last_historical['Total_Sales'], first_forecast['Forecasted_Sales']],
            mode='lines',
            name='Connection',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add a special legend entry for closed days to make them clear
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Closed Days (Red Dots)',
        marker=dict(size=12, color='red', line=dict(width=2, color='white')),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Calculate date ranges for interactive controls
    all_dates = []
    if len(historical_data) > 0:
        all_dates.extend(historical_data['Operational Date'].tolist())
    if len(forecast_data) > 0:
        all_dates.extend(forecast_data['Operational Date'].tolist())
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Calculate default zoom based on time period selection
        if time_period == "Last 1 Month + Forecast":
            if len(historical_data) > 0:
                last_historical_date = historical_data['Operational Date'].max()
                default_start = last_historical_date - pd.Timedelta(days=30)
            else:
                default_start = min_date
            default_end = max_date
        elif time_period == "Last 2 Weeks + Forecast":
            if len(historical_data) > 0:
                last_historical_date = historical_data['Operational Date'].max()
                default_start = last_historical_date - pd.Timedelta(days=14)
            else:
                default_start = min_date
            default_end = max_date
        elif time_period == "Last 1 Week + Forecast":
            if len(historical_data) > 0:
                last_historical_date = historical_data['Operational Date'].max()
                default_start = last_historical_date - pd.Timedelta(days=7)
            else:
                default_start = min_date
            default_end = max_date
        elif time_period == "Forecast Only":
            if len(forecast_data) > 0:
                default_start = forecast_data['Operational Date'].min()
                default_end = forecast_data['Operational Date'].max()
            else:
                default_start = min_date
                default_end = max_date
        else:  # "All Data"
            default_start = min_date
            default_end = max_date
    else:
        default_start = pd.Timestamp.now() - pd.Timedelta(days=30)
        default_end = pd.Timestamp.now()
    
    # Enhanced layout with interactive controls
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=True, thickness=0.1),  # Add range slider
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=14, label="14D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            range=[default_start, default_end],  # Set default zoom
            type='date',
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Sales ($)',
            fixedrange=False,  # Allow zooming on Y-axis too
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        template='plotly_white',
        height=chart_height,  # User-controlled height
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0, y=1, 
            bgcolor='rgba(255,255,255,0.8)', 
            font=dict(size=10),
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=40, r=20, t=60, b=80),  # More space for controls
        # Add crossfilter for better interactivity
        dragmode='zoom'
    )
    
    # Enhanced interactive configuration
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'modeBarButtonsToAdd': [
            'drawline', 'drawopenpath', 'drawclosedpath', 
            'drawcircle', 'drawrect', 'eraseshape'
        ],
        'toImageButtonOptions': {
            'format': 'png', 
            'filename': 'sales_forecast', 
            'height': 600,
            'width': 1200,
            'scale': 2
        },
        'scrollZoom': True,  # Enable scroll to zoom
        'doubleClick': 'reset+autosize',  # Double-click to reset zoom
        'showTips': True,
        'responsive': True
    }
    
    return fig, config

# Simplified line chart creation
@st.cache_data  
def create_line_chart(data, x, y, title, xlabel, ylabel, markers=True, color=None):
    """Create a simple line chart with caching"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode='lines+markers' if markers else 'lines',
        name=y,
        line=dict(color=color or '#1f77b4', width=2),
        marker=dict(size=4) if markers else None,
        hovertemplate=f'<b>%{{x}}</b><br>{ylabel}: %{{y:,.0f}}<extra></extra>'
    ))
    
    # Calculate date range for controls
    if len(data) > 0:
        min_date = data[x].min()
        max_date = data[x].max()
        # Default to show last 30 days if more data available
        if (max_date - min_date).days > 30:
            default_start = max_date - pd.Timedelta(days=30)
            default_end = max_date
        else:
            default_start = min_date
            default_end = max_date
    else:
        default_start = pd.Timestamp.now() - pd.Timedelta(days=30)
        default_end = pd.Timestamp.now()
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title=xlabel,
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=14, label="14D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            range=[default_start, default_end],
            type='date'
        ),
        yaxis=dict(
            title=ylabel,
            fixedrange=False
        ),
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=20, t=60, b=80),
        dragmode='zoom'
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
        'modeBarButtonsToAdd': [
            'drawline', 'drawopenpath', 'drawclosedpath', 
            'drawcircle', 'drawrect', 'eraseshape'
        ],
        'toImageButtonOptions': {
            'format': 'png', 
            'filename': 'sales_chart', 
            'height': 600,
            'width': 1200,
            'scale': 2
        },
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'showTips': True,
        'responsive': True
    }
    
    return fig, config

# Optimized feature importance calculation
@st.cache_data
def get_feature_importance(_model_results, features, selected_model):
    """Get feature importance from the selected model - cached"""
    try:
        model = _model_results[selected_model]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame(columns=['Feature', 'Importance'])
            
    except Exception as e:
        st.error(f"❌ Error getting feature importance: {str(e)}")
        return pd.DataFrame(columns=['Feature', 'Importance'])

if __name__ == "__main__":
    main()