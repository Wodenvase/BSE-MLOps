# SENSEX MLOps Pipeline - Phase 1: Automated Data Engineering

## 🎯 Overview

Phase 1 of the SENSEX MLOps pipeline focuses on building a **reliable, scheduled data engineering pipeline** that automatically fetches data for all 30 SENSEX components, processes them into feature maps, and versions the data using DVC.

### Key Objectives
- ✅ **Automated Data Collection**: Scrape SENSEX 30 component tickers and fetch 2-year historical data
- ✅ **Feature Engineering**: Transform raw OHLCV data into (num_days, 30, k_features) feature maps
- ✅ **Pipeline Orchestration**: Use Apache Airflow for reliable scheduling and monitoring
- ✅ **Data Versioning**: Implement DVC with Google Drive for data version control
- ✅ **Production Ready**: Docker deployment with comprehensive monitoring and error handling

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Ticker Scraping   │    │   Data Fetching     │    │ Feature Processing  │
│                     │────▶                     │────▶                     │
│ • BSE Website       │    │ • yfinance API      │    │ • 40+ Tech Indicators│
│ • MoneyControl      │    │ • Parallel Processing│   │ • Feature Selection │
│ • Investing.com     │    │ • Data Validation   │    │ • Normalization     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Apache Airflow Orchestration                        │
│ • Task Dependencies  • Error Handling  • Retry Logic  • Monitoring         │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Versioning   │    │   Storage & Backup  │    │    Monitoring       │
│                     │    │                     │    │                     │
│ • DVC Integration   │    │ • Google Drive      │    │ • Airflow UI        │
│ • Git Integration   │    │ • Local Storage     │    │ • MLflow Tracking   │
│ • Version Tags      │    │ • Docker Volumes    │    │ • Streamlit Dashboard│
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 Project Structure

```
sensex-mlops/
├── 🚀 Phase 1: Data Engineering Pipeline
│   ├── dags/
│   │   └── phase1_data_engineering_dag.py    # Main Airflow DAG
│   ├── src/
│   │   ├── data/
│   │   │   ├── get_sensex_tickers.py         # Multi-source ticker scraper
│   │   │   ├── fetch_data.py                 # Enhanced data fetcher
│   │   │   └── process_features.py           # Advanced feature processor
│   │   └── utils/
│   │       └── dvc_manager.py                # DVC operations manager
│   ├── data/
│   │   ├── raw/                              # Raw OHLCV data
│   │   └── processed/                        # Feature maps & targets
│   ├── .dvc/
│   │   └── config                            # DVC configuration
│   ├── docker-compose.yml                    # Multi-service deployment
│   ├── Dockerfile.airflow                    # Custom Airflow image
│   ├── deploy.sh                             # Automated deployment script
│   └── requirements.txt                      # Python dependencies
├── 📊 MLOps Infrastructure
│   ├── mlruns/                               # MLflow tracking
│   ├── streamlit_app/                        # Dashboard application
│   └── configs/                              # Configuration files
└── 📝 Documentation
    ├── README.md                             # This file
    └── docs/                                 # Additional documentation
```

## 🔧 Component Details

### 1. SENSEX Ticker Scraper (`get_sensex_tickers.py`)
**Multi-source scraper with fallback mechanisms**

```python
# Key Features:
- 🎯 Scrapes from BSE, MoneyControl, Investing.com
- 🔄 Automatic fallback if one source fails
- ✅ yfinance validation of ticker symbols
- 💾 Caching to reduce API calls
- 📋 Returns 30 validated .NS symbols
```

**Usage:**
```python
from data.get_sensex_tickers import SensexTickerScraper

scraper = SensexTickerScraper()
tickers = scraper.get_sensex_components(validate=True)
# Returns: ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', ...]
```

### 2. Enhanced Data Fetcher (`fetch_data.py`)
**Robust data fetching with parallel processing**

```python
# Key Features:
- 🚀 Parallel fetching with ThreadPoolExecutor
- 🔄 Exponential backoff retry logic
- ✅ Data quality validation and reporting
- 💾 Automatic data saving and loading
- 📊 Comprehensive error handling
```

**Capabilities:**
- Fetches 2 years of daily OHLCV data
- Processes 30+ stocks in parallel
- Validates data completeness and quality
- Generates detailed quality reports

### 3. Advanced Feature Processor (`process_features.py`)
**Transforms raw data into ML-ready feature maps**

```python
# Key Features:
- 📈 40+ Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
- 🎯 Feature Selection (variance, correlation-based)
- 📊 Multiple normalization methods
- 🔄 Handles missing data intelligently
- 📐 Outputs (num_days, 30, k_features) format
```

**Feature Categories:**
- **Price Features**: Returns, volatility, price ratios
- **Technical Indicators**: RSI, MACD, Stochastic, Williams %R
- **Volume Features**: Volume ratios, volume-price trends
- **Volatility Features**: ATR, Bollinger Band positions
- **Momentum Features**: Rate of change, momentum oscillators

### 4. Production Airflow DAG (`phase1_data_engineering_dag.py`)
**Comprehensive pipeline orchestration**

```python
# Pipeline Flow:
Setup → Health Check → Ticker Scraping → Data Fetching → 
Feature Processing → Data Versioning → Success Notification → Cleanup
```

**Key Features:**
- 🏗️ **Task Groups**: Organized, maintainable task structure
- 🔄 **Retry Logic**: Exponential backoff with 2 retries
- ✅ **Validation**: Data quality checks at each step
- 📊 **Monitoring**: Comprehensive logging and metrics
- ⏰ **Scheduling**: Runs at 6 AM on weekdays
- 🚨 **Error Handling**: Graceful failure handling

### 5. DVC Integration (`dvc_manager.py`)
**Automated data version control**

```python
# Key Features:
- 🌐 Google Drive remote storage
- 🔄 Automated data tracking
- 📝 Version tagging and metadata
- 📊 Data status monitoring
- 🔧 Easy remote configuration
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Git
- 8GB+ available disk space
- 4GB+ RAM

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sensex-mlops
```

### 2. Deploy the Pipeline
```bash
# Automated deployment (recommended)
./deploy.sh

# Or step-by-step deployment
./deploy.sh setup    # Setup environment
./deploy.sh build    # Build Docker images  
./deploy.sh start    # Start services
```

### 3. Configure Google Drive (Optional but Recommended)
1. Create a Google Service Account
2. Download the JSON credentials
3. Replace `google-credentials.json` with your credentials
4. Get your Google Drive folder ID and set it:
```bash
export GOOGLE_DRIVE_FOLDER_ID="your-folder-id-here"
```

### 4. Access the Pipeline
- **Airflow Dashboard**: http://localhost:8080 (admin/admin123)
- **MLflow Tracking**: http://localhost:5000
- **Streamlit Dashboard**: http://localhost:8501
- **Jupyter Notebook**: http://localhost:8888

### 5. Run the Pipeline
1. Go to Airflow Dashboard
2. Enable the `sensex_data_engineering_pipeline_v1` DAG
3. Trigger manually or wait for scheduled run (6 AM weekdays)

## 📊 Pipeline Outputs

### Raw Data (`data/raw/`)
```
RELIANCE.NS.csv    # Raw OHLCV data for each stock
TCS.NS.csv
HDFCBANK.NS.csv
...
^BSESN.csv         # SENSEX index data
```

### Processed Data (`data/processed/`)
```
feature_maps.npy      # Shape: (num_days, 30, k_features)
targets.npy           # Shape: (num_days,) - Binary classification targets
dates.csv             # Corresponding dates
feature_metadata.json # Feature names and metadata
pipeline_metadata.json # Pipeline execution metadata
```

### DVC Tracking
```
feature_maps.npy.dvc  # DVC tracking files
targets.npy.dvc
feature_metadata.json.dvc
```

## 🔍 Monitoring and Troubleshooting

### Check Pipeline Status
```bash
./deploy.sh status                    # Check all services
./deploy.sh logs airflow-scheduler    # Check specific service logs
docker-compose ps                     # Docker container status
```

### Common Issues and Solutions

#### 1. Airflow Services Not Starting
```bash
# Check logs
./deploy.sh logs airflow-webserver

# Restart services
./deploy.sh restart

# Rebuild if needed
docker-compose down
docker-compose build --no-cache
./deploy.sh start
```

#### 2. DVC Authentication Issues
```bash
# Re-authenticate with Google Drive
dvc auth login

# Check remote configuration
dvc remote list -v
```

#### 3. Data Pipeline Failures
1. Check Airflow UI for failed tasks
2. View task logs in Airflow
3. Check data quality reports in `logs/data_quality_report.json`
4. Verify ticker availability and API limits

### Performance Monitoring
- **Airflow UI**: Task duration, success rates, retry patterns
- **MLflow**: Data quality metrics, pipeline performance
- **System Resources**: CPU, memory, disk usage via Docker stats

## 🔧 Configuration

### Environment Variables
```bash
# Airflow Configuration
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123

# Data Pipeline
SENSEX_DATA_PERIOD=2y          # Data collection period
SENSEX_MIN_STOCKS=25           # Minimum stocks required
SENSEX_TARGET_FEATURES=40      # Target number of features

# Google Drive Integration
GOOGLE_DRIVE_FOLDER_ID=your-folder-id
```

### Pipeline Configuration
Modify `get_pipeline_config()` in the DAG file:
```python
{
    'data_period': '2y',           # Historical data period
    'min_stocks_required': 25,     # Minimum stocks for pipeline success
    'target_features': 40,         # Number of features to select
    'feature_selection_method': 'variance',  # Feature selection method
    'parallel_fetching': True,     # Enable parallel data fetching
    'max_workers': 8,              # Number of parallel workers
    'data_quality_threshold': 0.8  # Minimum data quality score
}
```

## 📈 Data Quality Metrics

The pipeline tracks several data quality metrics:

### Fetching Metrics
- **Success Rate**: Percentage of successfully fetched stocks
- **Data Completeness**: Percentage of expected data points received
- **Data Freshness**: Age of the most recent data point
- **API Response Times**: Average response time per stock

### Processing Metrics
- **Feature Coverage**: Percentage of features successfully calculated
- **Missing Data Handling**: Percentage of missing values imputed
- **Normalization Quality**: Distribution statistics after normalization
- **Target Balance**: Class distribution for binary targets

### Example Quality Report
```json
{
  "timestamp": "2024-01-15T06:30:00",
  "stocks_requested": 30,
  "stocks_successful": 29,
  "success_rate": 0.967,
  "data_completeness": {
    "RELIANCE.NS": 0.995,
    "TCS.NS": 1.000,
    "avg_completeness": 0.987
  },
  "feature_quality": {
    "total_features": 42,
    "valid_features": 40,
    "missing_data_percentage": 0.02
  }
}
```

## 🔄 Next Steps: Phase 2 Preview

Phase 1 establishes the data foundation. Phase 2 will focus on:

1. **ConvLSTM Model Development**
   - Architecture design for time series forecasting
   - Hyperparameter optimization with Optuna
   - Model validation and backtesting

2. **Advanced MLOps Features**
   - Model training pipelines
   - A/B testing framework
   - Performance monitoring
   - Automated retraining

3. **Production Deployment**
   - Real-time inference API
   - Model serving with versioning
   - Monitoring and alerting
   - CI/CD integration

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive logging and error handling
3. Include unit tests for new components
4. Update documentation for any changes
5. Test with the deployment script before submitting

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Airflow logs and MLflow metrics
3. Check GitHub issues for similar problems
4. Create a new issue with detailed logs and environment info

---

**📊 Phase 1 Status: COMPLETED ✅**
- ✅ Automated ticker scraping with multi-source fallback
- ✅ Robust data fetching with parallel processing and retry logic
- ✅ Advanced feature processing with 40+ technical indicators
- ✅ Production Airflow DAG with comprehensive error handling
- ✅ DVC integration for automated data versioning
- ✅ Docker deployment with full monitoring stack
- ✅ Comprehensive documentation and deployment scripts

**🚀 Ready for Phase 2: ConvLSTM Model Development**
