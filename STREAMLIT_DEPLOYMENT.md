# Streamlit Dashboard Deployment Guide

## Quick Start

### 1. Local Development
```bash
# Run the deployment script
./deploy_streamlit.sh local
```

Or manually:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 2. Streamlit Cloud Deployment

#### Prerequisites
- GitHub repository (already set up at https://github.com/Wodenvase/BSE-MLOps)
- Streamlit Cloud account

#### Steps
1. **Prepare deployment files** (already done):
   ```bash
   ./deploy_streamlit.sh cloud
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Wodenvase/BSE-MLOps`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

#### Deployment Files Created
- `runtime.txt` - Specifies Python 3.9
- `packages.txt` - System dependencies
- `requirements.txt` - Python dependencies

### 3. Docker Deployment
```bash
# Build and run with Docker
./deploy_streamlit.sh docker
```

Or manually:
```bash
# Build image
docker build -t bse-mlops-streamlit .

# Run container
docker run -p 8501:8501 bse-mlops-streamlit
```

### 4. Other Cloud Platforms

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   git add .
   git commit -m "Add Heroku deployment"
   git push heroku main
   ```

#### Railway
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

#### Render
1. Connect GitHub repository
2. Environment: Python 3
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   - The app automatically falls back to demo mode if MLflow/yfinance aren't available
   - This is expected for cloud deployments

2. **Python Version Compatibility**
   - Ensure Python 3.9+ is used (specified in runtime.txt)
   - Type annotations require Python 3.9+

3. **Memory Issues**
   - Streamlit Cloud has memory limits
   - App includes memory optimization and caching

4. **Port Issues**
   - Use environment variable PORT for cloud platforms
   - Default is 8501 for local development

### Environment Variables

For production deployment, you can set:
- `STREAMLIT_SERVER_PORT` - Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Server address (default: 0.0.0.0)
- `MLFLOW_TRACKING_URI` - MLflow server (if available)

## Demo Mode

The app includes a comprehensive demo mode that:
- Simulates realistic SENSEX predictions
- Shows market data visualization
- Demonstrates full UI/UX capabilities
- Works without external dependencies

This ensures the app works reliably across all deployment platforms.
