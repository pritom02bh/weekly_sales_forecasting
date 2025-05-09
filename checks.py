#!/usr/bin/env python
"""
Pre-deployment validation script for the Sales Forecasting application.
Run this script before deploying to validate your environment.
"""

import os
import sys
import importlib
import pathlib
import pandas as pd

def check_imports():
    """Check if all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'streamlit', 'plotly', 'joblib', 'statsmodels', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing_packages

def check_data_files():
    """Check if data files exist in the expected location."""
    # Get the path to the data directory
    base_path = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(base_path, "data")
    
    # List of possible data files
    data_files = [
        'march_data_complete.csv',
        'april_data.csv',
        'historical_march_data.csv',
        'april_first_week.csv',
        'forecasting_data_march.csv',
        'april_weather.csv'
    ]
    
    found_files = []
    missing_files = []
    
    # Check if the data directory exists
    if not os.path.exists(data_path):
        print(f"❌ Data directory not found at {data_path}")
        return False
    
    # Check for at least one set of required data files
    for file_name in data_files:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            found_files.append(file_name)
            print(f"✅ Found data file: {file_name}")
        else:
            missing_files.append(file_name)
    
    # We need at least one March data file and one April data file
    march_files = [f for f in found_files if 'march' in f.lower()]
    april_files = [f for f in found_files if 'april' in f.lower()]
    
    if not march_files:
        print("❌ No March data files found")
        return False
    
    if not april_files:
        print("❌ No April data files found")
        return False
    
    print(f"✅ Found {len(found_files)} data files")
    return True

def check_streamlit_config():
    """Check if Streamlit configuration files exist."""
    base_path = pathlib.Path(__file__).parent.absolute()
    config_path = os.path.join(base_path, ".streamlit", "config.toml")
    
    if os.path.exists(config_path):
        print(f"✅ Streamlit config found at {config_path}")
        return True
    else:
        print(f"❌ Streamlit config not found at {config_path}")
        return False

def check_deployment_files():
    """Check if deployment-related files exist."""
    base_path = pathlib.Path(__file__).parent.absolute()
    files_to_check = {
        'requirements.txt': 'Package dependencies',
        'Procfile': 'Process file for web deployment',
        'runtime.txt': 'Python runtime specification',
        '.gitignore': 'Git ignore file'
    }
    
    for file_name, description in files_to_check.items():
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ Found {file_name} ({description})")
        else:
            print(f"❌ Missing {file_name} ({description})")

def check_models_directory():
    """Check if the models directory exists or can be created."""
    base_path = pathlib.Path(__file__).parent.absolute()
    models_path = os.path.join(base_path, "models")
    
    if os.path.exists(models_path):
        print(f"✅ Models directory exists at {models_path}")
        # Check if it's writable
        if os.access(models_path, os.W_OK):
            print("✅ Models directory is writable")
            return True
        else:
            print("❌ Models directory is not writable")
            return False
    else:
        print(f"ℹ️ Models directory does not exist at {models_path}")
        try:
            os.makedirs(models_path)
            print(f"✅ Created models directory at {models_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create models directory: {e}")
            return False

def run_all_checks():
    """Run all pre-deployment checks."""
    print("\n==== Running Pre-deployment Checks ====\n")
    
    # Check imports
    print("\n--- Checking Required Packages ---")
    missing_packages = check_imports()
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are installed")
    
    # Check data files
    print("\n--- Checking Data Files ---")
    data_files_ok = check_data_files()
    
    # Check Streamlit config
    print("\n--- Checking Streamlit Configuration ---")
    streamlit_config_ok = check_streamlit_config()
    
    # Check deployment files
    print("\n--- Checking Deployment Files ---")
    check_deployment_files()
    
    # Check models directory
    print("\n--- Checking Models Directory ---")
    models_directory_ok = check_models_directory()
    
    # Overall status
    print("\n==== Check Results ====")
    if not missing_packages and data_files_ok and streamlit_config_ok and models_directory_ok:
        print("\n✅ All checks passed! Your application is ready for deployment.")
    else:
        print("\n⚠️ Some checks failed. Please address the issues before deploying.")

if __name__ == "__main__":
    run_all_checks() 