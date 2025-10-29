#!/bin/bash

# Streamlit Deployment Script for BSE MLOps Project
# This script helps deploy the Streamlit dashboard to various platforms

echo "🚀 BSE MLOps Streamlit Deployment Script"
echo "========================================="

# Function to display usage
usage() {
    echo "Usage: $0 [local|cloud|docker]"
    echo ""
    echo "Options:"
    echo "  local   - Run locally"
    echo "  cloud   - Deploy to Streamlit Cloud"
    echo "  docker  - Build and run with Docker"
    echo ""
    exit 1
}

# Function to run locally
run_local() {
    echo "🏠 Running Streamlit app locally..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    # Install requirements
    echo "📥 Installing requirements..."
    pip install -r requirements.txt
    
    # Run streamlit app
    echo "🎯 Starting Streamlit app..."
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
}

# Function to prepare for cloud deployment
prepare_cloud() {
    echo "☁️  Preparing for Streamlit Cloud deployment..."
    
    # Check if runtime.txt exists
    if [ ! -f "runtime.txt" ]; then
        echo "📝 Creating runtime.txt..."
        echo "python-3.9" > runtime.txt
    fi
    
    # Check if packages.txt exists
    if [ ! -f "packages.txt" ]; then
        echo "📝 Creating packages.txt..."
        cat > packages.txt << EOF
build-essential
python3-dev
EOF
    fi
    
    echo "✅ Cloud deployment files ready!"
    echo ""
    echo "📋 Next steps for Streamlit Cloud deployment:"
    echo "1. Push your code to GitHub"
    echo "2. Go to https://share.streamlit.io/"
    echo "3. Connect your GitHub repository"
    echo "4. Deploy with the following settings:"
    echo "   - Repository: Wodenvase/BSE-MLOps"
    echo "   - Branch: main"
    echo "   - Main file path: app.py"
    echo ""
}

# Function to run with Docker
run_docker() {
    echo "🐳 Building and running with Docker..."
    
    # Build Docker image
    echo "🔨 Building Docker image..."
    docker build -t bse-mlops-streamlit .
    
    # Run Docker container
    echo "🚀 Running Docker container..."
    docker run -p 8501:8501 bse-mlops-streamlit
}

# Main script logic
case "${1:-}" in
    local)
        run_local
        ;;
    cloud)
        prepare_cloud
        ;;
    docker)
        run_docker
        ;;
    *)
        usage
        ;;
esac
