# 📊 Weekly Sales Forecasting

> **Machine learning sales forecasting system** with weather integration and interactive dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏢 **Multi-Location** | 4 Dutch retail locations |
| 🤖 **ML Models** | Gradient Boosting & XGBoost |
| 🌤️ **Weather Integration** | Weather data as predictive features |
| 📱 **Interactive Dashboard** | Streamlit web interface |
| 💾 **Model Persistence** | Versioning & metadata |
| 📈 **Forecasting** | Daily & weekly predictions |
| 📊 **Performance Metrics** | MAE, RMSE, R² score |
| ✅ **Data Validation** | Preprocessing & quality checks |

## 🏪 Business Locations

| Location | ID | Records |
|----------|----|---------:|
| **Fenix Food Factory B.V.** | 50460 | 412 |
| **Kaapse Will'ns B.V.** | 47904 | 302 |
| **Kaapse Maria B.V.** | 47903 | 365 |
| **Kaapse Kaap B.V.** | 47901 | 94 |

## 📁 Project Structure

```bash
weekly_sales_forecasting/
├── 📊 streamlit_app.py                    # Main dashboard (2,835 lines)
├── 🔬 sales_forecasting.py               # Core algorithms (234 lines)  
├── ✅ checks.py                           # Validation (170 lines)
├── 📋 requirements.txt                    # Dependencies
├── 🚀 Procfile                            # Deployment
├── 🐍 runtime.txt                         # Python version
├── ⚙️  .streamlit/config.toml              # App configuration
├── 📂 data/                               # Data files
│   ├── All_Locations_Combined.csv        # 📊 Combined (477 records)
│   ├── Fenix_Food_Factory_B.V._50460.csv # 🏭 Factory (412 records)
│   ├── Kaapse_Will'ns_B.V._47904.csv     # 🏪 Will'ns (302 records)
│   ├── Kaapse_Maria_B.V._47903.csv       # 🏪 Maria (365 records)
│   ├── Kaapse_Kaap_B.V._47901.csv        # 🏪 Kaap (94 records)
│   ├── Weather_May15_to_May27.csv        # 🌤️ Recent weather
│   ├── Weather_Kaapse_Kaap_47901.csv     # 🌤️ Location weather
│   └── dataset details.pdf               # 📖 Documentation
└── 💾 models/                             # Model storage
    ├── model_combined_gradient_boosting_*.pkl
    ├── model_combined_xgboost_*.pkl
    └── Versioned with timestamp & feature hash
```


## 🗃️ Data Structure

<details>
<summary><strong>📊 Sales Data Schema</strong></summary>

| Field | Type | Description |
|-------|------|-------------|
| `Operational Date` | Date | Daily timestamp |
| `Total_Sales` | Float | Revenue amount |
| `Sales_Count` | Integer | Transaction count |
| `Day_of_Week` | String | Monday-Sunday |
| `Is_Weekend` | Boolean | Weekend flag |
| `Is_Public_Holiday` | Boolean | Holiday flag |
| `Is_Closed` | Boolean | Store closure |
| `Tips_per_Transaction` | Float | Average tips |
| `Avg_Sale_per_Transaction` | Float | Average sale value |

</details>

<details>
<summary><strong>🌤️ Weather Data Schema</strong></summary>

| Field | Type | Description |
|-------|------|-------------|
| `tempmax/tempmin/temp` | Float | Temperature (°F) |
| `humidity` | Float | Humidity % |
| `precip` | Float | Precipitation |
| `precipprob` | Float | Precipitation % |
| `cloudcover` | Float | Cloud coverage % |
| `solarradiation` | Float | Solar radiation |
| `uvindex` | Float | UV index |

</details>

## 🤖 Machine Learning

```mermaid
graph LR
    A[📊 Data] --> B[🔧 Features]
    B --> C[🎯 Models]
    C --> D[📈 Forecasts]
    
    B1[⏰ Temporal] --> B
    B2[🌤️ Weather] --> B
    B3[💰 Sales] --> B
    B4[🏪 Location] --> B
    
    C1[🌳 Gradient Boosting] --> C
    C2[⚡ XGBoost] --> C
```

| Component | Details |
|-----------|---------|
| **🎯 Models** | Gradient Boosting, XGBoost |
| **🔧 Features** | Temporal, weather, sales patterns, location-specific |
| **✅ Validation** | Cross-validation with MAE, RMSE, R² |
| **💾 Management** | Versioning, compression, caching, cleanup |

## 📱 Dashboard

| Section | Features |
|---------|----------|
| **🎛️ Interface** | Company selection, date range, model config |
| **📊 Visualization** | Historical trends, forecasts, metrics |
| **📈 Analytics** | Time series, model comparison, reports |

## 🛠️ Technical Stack

| Category | Technologies |
|----------|--------------|
| **🖥️ Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **🤖 ML** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat) |
| **📊 Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **📈 Visualization** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) |
| **💾 Storage** | Joblib, compressed models with metadata |

## 📋 File Details

| File | Purpose | Key Features |
|------|---------|--------------|
| **📊 streamlit_app.py** | Main dashboard | Caching, data management, ML training, visualization |
| **🔬 sales_forecasting.py** | Core algorithms | Statistical analysis, model training, evaluation |
| **✅ checks.py** | Validation system | Dependencies, data integrity, deployment testing |

## 🚀 Deployment

<div align="center">

**Ready for Deployment:**

[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/cloud)

</div>

**Configuration files:** `Procfile` • `runtime.txt` • `requirements.txt` • `.streamlit/config.toml`

---

<div align="center">
<strong>📊 Sales Forecasting Dashboard</strong><br>
<em>Powered by Machine Learning & Weather Intelligence</em>
</div> 
