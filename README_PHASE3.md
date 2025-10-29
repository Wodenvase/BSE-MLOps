# Phase 3: Model Serving & Interactive Application

## 🎯 Overview

Phase 3 completes our SENSEX MLOps pipeline by creating a production-ready interactive web application that serves our best ConvLSTM model for real-time SENSEX next-day movement predictions.

## 🏗️ Architecture

### Core Components

1. **MLflow Model Registry** (`src/serving/model_registry.py`)
   - Manages model versions and promotion workflow
   - Handles Production/Staging/Archived model stages
   - Provides automated best model selection and promotion

2. **Model Server** (`src/serving/model_server.py`)
   - Production model serving infrastructure
   - Real-time prediction pipeline with caching
   - Health monitoring and logging capabilities

3. **Real-time Data Pipeline** (`src/serving/realtime_data.py`)
   - Live market data fetching for SENSEX components
   - Data quality validation and preprocessing
   - Multi-threaded data collection with caching

4. **Interactive Streamlit App** (`streamlit_app/app.py`)
   - User-friendly web interface for predictions
   - Real-time market overview and visualization
   - Model performance monitoring dashboard

## 🚀 Quick Start

### 1. Setup Model Registry

```bash
# Navigate to serving directory
cd src/serving

# Setup MLflow Model Registry
python model_registry.py
```

This will:
- Create registered model in MLflow
- Find best performing model from Phase 2 experiments
- Promote best model to Production stage

### 2. Launch Interactive Application

```bash
# Navigate to streamlit app directory
cd streamlit_app

# Launch the application
./run_app.sh
```

The app will be available at: http://localhost:8501

## 🎮 Using the Application

### Main Features

1. **Next-Day Prediction**
   - Click "🔮 Run Prediction" to generate forecast
   - View probability, direction (UP/DOWN), and confidence
   - Risk assessment based on model confidence

2. **Market Overview**
   - Real-time SENSEX index price
   - Market breadth (advancing vs declining stocks)
   - Top gainers and losers

3. **System Monitoring**
   - Model health checks
   - Data pipeline status
   - Prediction history tracking

### Prediction Workflow

1. **Data Fetching**: Latest 60 days of SENSEX component data
2. **Feature Processing**: 40+ technical indicators calculated
3. **Model Inference**: ConvLSTM prediction with probability scores
4. **Result Display**: Direction, confidence, and risk assessment

## 📊 Model Registry Management

### Automated Promotion Workflow

```python
from src.serving.model_registry import ModelRegistry

registry = ModelRegistry()

# Setup automated promotion (finds best model)
registry.setup_automated_promotion()

# Check production model
prod_model = registry.get_production_model()
print(f"Production Model Version: {prod_model['version']}")
```

### Manual Model Management

```python
# List all model versions
versions = registry.list_all_versions()

# Promote specific version
registry.promote_model_to_production(version="3", min_accuracy=0.55)

# Evaluate model drift
drift_analysis = registry.evaluate_model_drift(current_metrics)
```

## 🔧 Configuration

### Environment Variables

```bash
# MLflow tracking (optional, defaults to local)
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Server Configuration

```python
# Custom configuration
server = ModelServer()
server.cache_ttl = 1800  # 30 minutes cache
server.load_production_model()
```

## 📈 Real-time Data Pipeline

### Data Sources
- **Primary**: Yahoo Finance API (yfinance)
- **Symbols**: All 30 SENSEX components + SENSEX index
- **Update Frequency**: 5-minute cache for prices, 1-hour for historical data

### Data Quality Checks
- Minimum 30 days of data per symbol
- At least 25/30 symbols must have valid data
- Automatic retry logic with exponential backoff
- Data freshness validation (alerts if >24 hours old)

### Performance Optimizations
- Multi-threaded data fetching (10 concurrent threads)
- Intelligent caching with TTL management
- Batch API calls where possible
- Graceful fallback for individual symbol failures

## 🏥 Monitoring & Health Checks

### Model Server Health

```python
health_status = server.health_check()
{
    'status': 'healthy',
    'checks': {
        'model_loaded': True,
        'model_inference': True,
        'cache_healthy': True
    }
}
```

### Data Pipeline Health

```python
quality_report = fetcher.validate_data_quality(data)
{
    'status': 'valid',
    'readiness': True,
    'statistics': {
        'valid_symbols': 28,
        'data_coverage': 93.3
    }
}
```

## 🔐 Production Considerations

### Security
- No sensitive data stored in application
- API rate limiting through caching
- Input validation for all user inputs

### Performance
- Model inference: ~100-200ms
- Data fetching: ~2-5 seconds (with caching: ~50ms)
- Memory usage: ~500MB-1GB depending on cache size

### Scalability
- Stateless design allows horizontal scaling
- Redis can replace in-memory caching for multi-instance deployment
- Model server can be containerized and deployed separately

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 📝 Logging & Monitoring

### Prediction Logging
All predictions are logged to `prediction_log.json`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "prediction": {
    "direction": "UP",
    "probability": 0.67,
    "confidence": "Medium"
  },
  "model_version": "3"
}
```

### Model Promotion Logging
Model promotions logged to `model_promotion_log.json`:

```json
{
  "timestamp": "2024-01-15T09:00:00",
  "model_name": "sensex-convlstm-model",
  "version": "3",
  "stage": "Production",
  "event_type": "promotion"
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Loading**
   ```
   Error: No production model available
   ```
   - Run model registry setup: `python model_registry.py`
   - Ensure Phase 2 training has been completed

2. **Data Fetching Errors**
   ```
   Error: Insufficient data for prediction
   ```
   - Check internet connection
   - Verify Yahoo Finance API is accessible
   - Clear cache and retry

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src.serving'
   ```
   - Ensure you're running from correct directory
   - Check PYTHONPATH includes project root

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH="../" streamlit run app.py --logger.level=debug
```

## 🔮 Future Enhancements

1. **Advanced Visualizations**
   - Feature importance heatmaps
   - SHAP value explanations
   - Technical indicator charts

2. **Model Ensemble**
   - Multiple model voting
   - Confidence-weighted predictions
   - Uncertainty quantification

3. **Real-time Alerts**
   - Email/SMS notifications
   - Webhook integrations
   - Custom alert conditions

4. **Extended Market Coverage**
   - Multiple indices (NIFTY, international)
   - Sector-specific predictions
   - Individual stock forecasts

## 📊 Performance Metrics

### Application Performance
- **Cold Start**: ~5-10 seconds (model loading)
- **Warm Predictions**: ~1-2 seconds
- **Data Refresh**: ~3-5 seconds
- **Cache Hit Rate**: >90% (typical usage)

### Model Performance
- **Accuracy**: 55-65% (binary classification)
- **Precision**: 0.52-0.62
- **Recall**: 0.48-0.58
- **AUC**: 0.55-0.65

## 🎓 Educational Value

This Phase 3 implementation demonstrates:

1. **MLOps Best Practices**
   - Model registry and versioning
   - Automated promotion workflows
   - Health monitoring and logging

2. **Production ML Serving**
   - Real-time inference pipelines
   - Data validation and quality checks
   - Caching and performance optimization

3. **Interactive ML Applications**
   - User-friendly interfaces for ML models
   - Real-time data integration
   - Comprehensive system monitoring

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `prediction_log.json` and `model_promotion_log.json`
3. Ensure all Phase 1 and Phase 2 components are working

---

🎉 **Congratulations!** You now have a complete end-to-end MLOps pipeline for SENSEX prediction, from data engineering (Phase 1) through model training (Phase 2) to production serving (Phase 3)!
