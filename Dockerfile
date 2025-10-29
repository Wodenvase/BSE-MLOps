# Production Dockerfile for SENSEX Next-Day Forecast Streamlit App
# Optimized for serving, not training

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app/ /app/streamlit_app/
COPY src/ /app/src/
COPY README_PHASE3.md /app/README_PHASE3.md

# Create necessary directories
RUN mkdir -p /app/models /app/data/processed /app/logs /app/mlruns

# Create a lightweight model placeholder (for demo when MLflow is not available)
RUN echo '{"model_info": "demo", "version": "1.0.0", "status": "placeholder"}' > /app/models/model_info.json

# Set working directory to streamlit app
WORKDIR /app/streamlit_app

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
