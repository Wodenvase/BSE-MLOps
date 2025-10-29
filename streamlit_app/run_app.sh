#!/bin/bash

# SENSEX Next-Day Forecast - Streamlit App Launcher
# This script sets up and launches the interactive ML application

echo "🚀 Starting SENSEX Next-Day Forecast Application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    print_error "app.py not found. Please run this script from the streamlit_app directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ] && [ ! -d "venv" ]; then
    print_warning "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
else
    print_status "Activating virtual environment..."
    if [ -d "../venv" ]; then
        source ../venv/bin/activate
    else
        source venv/bin/activate
    fi
fi

# Install/upgrade requirements
print_status "Installing required packages..."
pip install -r requirements.txt

# Check if MLflow server is running (optional)
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    print_warning "MLflow server not detected at localhost:5000"
    print_warning "The app will use local model loading as fallback"
else
    print_success "MLflow server detected"
fi

# Create necessary directories
print_status "Setting up directories..."
mkdir -p ../models
mkdir -p ../data/processed
mkdir -p ../logs

# Set environment variables
export PYTHONPATH="../:$PYTHONPATH"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Check port availability
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null; then
    print_warning "Port 8501 is already in use"
    print_status "Trying alternative port 8502..."
    export STREAMLIT_SERVER_PORT=8502
fi

print_success "Environment setup complete!"
print_status "Starting Streamlit application..."

echo ""
echo "==============================================="
echo "📈 SENSEX Next-Day Forecast Application"
echo "==============================================="
echo "🔗 URL: http://localhost:${STREAMLIT_SERVER_PORT}"
echo "🛑 Press Ctrl+C to stop the application"
echo "==============================================="
echo ""

# Launch Streamlit
streamlit run app.py \
    --server.port=${STREAMLIT_SERVER_PORT} \
    --server.address=${STREAMLIT_SERVER_ADDRESS} \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --theme.base="light" \
    --theme.primaryColor="#1f77b4" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6"
