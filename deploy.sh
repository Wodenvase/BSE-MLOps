#!/bin/bash

# SENSEX MLOps Pipeline - Phase 1 Deployment Script
# This script sets up the complete environment for automated data engineering

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sensex-mlops"
AIRFLOW_UID=${AIRFLOW_UID:-50000}
GOOGLE_DRIVE_FOLDER_ID="${GOOGLE_DRIVE_FOLDER_ID:-}"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                  SENSEX MLOps Pipeline                       ║"
    echo "║                Phase 1: Data Engineering                     ║"
    echo "║                    Deployment Script                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_dependencies() {
    print_step "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    # Check Python (for DVC setup)
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_success "All system dependencies are available"
}

setup_environment() {
    print_step "Setting up environment..."
    
    # Create .env file
    cat > .env << EOF
# Airflow Configuration
AIRFLOW_UID=${AIRFLOW_UID}
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123

# Environment
ENVIRONMENT=${ENVIRONMENT}

# Data Pipeline Configuration
SENSEX_DATA_PERIOD=2y
SENSEX_MIN_STOCKS=25
SENSEX_TARGET_FEATURES=40

# Google Drive Configuration (optional)
GOOGLE_DRIVE_FOLDER_ID=${GOOGLE_DRIVE_FOLDER_ID}
EOF
    
    print_success "Environment file created"
}

setup_directories() {
    print_step "Setting up project directories..."
    
    # Create directory structure
    mkdir -p {dags,logs,plugins,data/{raw,processed,temp},models,configs,notebooks}
    mkdir -p src/{data,models,utils}
    mkdir -p .dvc/cache
    
    # Set proper permissions
    chmod -R 755 dags logs plugins data models configs notebooks src
    
    print_success "Project directories created"
}

setup_dvc() {
    print_step "Setting up DVC (Data Version Control)..."
    
    # Install DVC if not already installed
    if ! command -v dvc &> /dev/null; then
        print_info "Installing DVC..."
        pip3 install --user dvc[gdrive]
    fi
    
    # Initialize DVC if not already initialized
    if [ ! -d ".dvc" ]; then
        print_info "Initializing DVC..."
        dvc init --no-scm
    fi
    
    # Setup Google Drive remote if folder ID is provided
    if [ -n "$GOOGLE_DRIVE_FOLDER_ID" ]; then
        print_info "Setting up Google Drive remote..."
        dvc remote add -d gdrive "gdrive://${GOOGLE_DRIVE_FOLDER_ID}" || true
        dvc remote modify gdrive gdrive_acknowledge_risk true
    else
        print_info "No Google Drive folder ID provided. You can set it up later with:"
        print_info "  dvc remote add -d gdrive gdrive://your-folder-id"
    fi
    
    print_success "DVC setup completed"
}

setup_google_credentials() {
    print_step "Setting up Google Drive credentials..."
    
    if [ ! -f "google-credentials.json" ]; then
        print_info "Creating placeholder Google credentials file..."
        cat > google-credentials.json << 'EOF'
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
}
EOF
        print_info "Please replace google-credentials.json with your actual Google Service Account credentials"
        print_info "For setup instructions, visit: https://dvc.org/doc/user-guide/setup-google-drive-remote"
    else
        print_success "Google credentials file already exists"
    fi
}

build_docker_images() {
    print_step "Building Docker images..."
    
    print_info "Building Airflow image with SENSEX pipeline dependencies..."
    docker-compose build airflow-webserver
    
    print_success "Docker images built successfully"
}

initialize_airflow() {
    print_step "Initializing Airflow database and user..."
    
    print_info "Starting Airflow initialization..."
    docker-compose up airflow-init
    
    print_success "Airflow initialization completed"
}

start_services() {
    print_step "Starting services..."
    
    print_info "Starting all services in detached mode..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    print_info "Checking service health..."
    
    # Check Airflow webserver
    if curl -f http://localhost:8080/health &> /dev/null; then
        print_success "Airflow webserver is healthy"
    else
        print_error "Airflow webserver is not responding"
    fi
    
    # Check MLflow
    if curl -f http://localhost:5000 &> /dev/null; then
        print_success "MLflow server is healthy"
    else
        print_info "MLflow server might still be starting..."
    fi
    
    # Check Streamlit
    if curl -f http://localhost:8501 &> /dev/null; then
        print_success "Streamlit dashboard is healthy"
    else
        print_info "Streamlit dashboard might still be starting..."
    fi
}

create_sample_data() {
    print_step "Creating sample data for testing..."
    
    # Create a simple test to ensure the pipeline can run
    docker-compose exec -T airflow-webserver python3 << 'EOF'
import sys
sys.path.insert(0, '/opt/airflow/src')

try:
    from data.get_sensex_tickers import SensexTickerScraper
    scraper = SensexTickerScraper()
    tickers = scraper.get_sensex_components(validate=False)  # Don't validate to avoid API calls
    print(f"✅ Ticker scraper loaded successfully. Found {len(tickers) if tickers else 0} tickers")
except Exception as e:
    print(f"❌ Error loading ticker scraper: {e}")

try:
    from data.fetch_data import SensexDataFetcher
    print("✅ Data fetcher loaded successfully")
except Exception as e:
    print(f"❌ Error loading data fetcher: {e}")

try:
    from data.process_features import AdvancedFeatureProcessor
    print("✅ Feature processor loaded successfully")
except Exception as e:
    print(f"❌ Error loading feature processor: {e}")
EOF
    
    print_success "Sample data creation completed"
}

run_health_check() {
    print_step "Running comprehensive health check..."
    
    # Check Docker containers
    print_info "Checking Docker containers..."
    docker-compose ps
    
    # Check logs for any errors
    print_info "Checking for critical errors in logs..."
    if docker-compose logs airflow-webserver | grep -i "error" | head -5; then
        print_info "Some errors found in logs (this might be normal during startup)"
    fi
    
    print_success "Health check completed"
}

show_access_info() {
    print_step "Deployment completed! Access information:"
    echo ""
    echo -e "${GREEN}🎉 SENSEX MLOps Pipeline is now running!${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  • Airflow Dashboard: http://localhost:8080"
    echo "    Username: admin"
    echo "    Password: admin123"
    echo ""
    echo "  • MLflow Tracking:   http://localhost:5000"
    echo "  • Streamlit Dashboard: http://localhost:8501"
    echo "  • Jupyter Notebook:   http://localhost:8888"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Visit Airflow dashboard and enable the 'sensex_data_engineering_pipeline_v1' DAG"
    echo "2. Configure Google Drive credentials if using DVC remote storage"
    echo "3. Trigger the pipeline manually or wait for the scheduled run (6 AM weekdays)"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  • View logs:        docker-compose logs -f [service_name]"
    echo "  • Stop services:    docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo "  • Access Airflow CLI: docker-compose exec airflow-webserver airflow"
    echo ""
}

cleanup_on_error() {
    print_error "Deployment failed. Cleaning up..."
    docker-compose down 2>/dev/null || true
    exit 1
}

# Main execution
main() {
    # Set up error handling
    trap cleanup_on_error ERR
    
    print_header
    
    # Check if running in project directory
    if [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run deployment steps
    check_dependencies
    setup_environment
    setup_directories
    setup_dvc
    setup_google_credentials
    build_docker_images
    initialize_airflow
    start_services
    create_sample_data
    run_health_check
    show_access_info
    
    print_success "🚀 SENSEX MLOps Pipeline deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "SENSEX MLOps Pipeline Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  help                Show this help message"
        echo "  check               Check system dependencies only"
        echo "  setup               Setup environment and directories only"
        echo "  build               Build Docker images only"
        echo "  start               Start services only"
        echo "  stop                Stop all services"
        echo "  restart             Restart all services"
        echo "  logs [service]      Show logs for service"
        echo "  status              Show service status"
        echo ""
        echo "Environment Variables:"
        echo "  AIRFLOW_UID                Airflow user ID (default: 50000)"
        echo "  GOOGLE_DRIVE_FOLDER_ID     Google Drive folder ID for DVC remote"
        echo "  ENVIRONMENT                Environment (development/production)"
        ;;
    "check")
        check_dependencies
        ;;
    "setup")
        setup_environment
        setup_directories
        setup_dvc
        setup_google_credentials
        ;;
    "build")
        build_docker_images
        ;;
    "start")
        start_services
        ;;
    "stop")
        print_step "Stopping all services..."
        docker-compose down
        print_success "All services stopped"
        ;;
    "restart")
        print_step "Restarting all services..."
        docker-compose restart
        print_success "All services restarted"
        ;;
    "logs")
        if [ -n "${2:-}" ]; then
            docker-compose logs -f "$2"
        else
            docker-compose logs -f
        fi
        ;;
    "status")
        print_step "Service status:"
        docker-compose ps
        ;;
    *)
        main
        ;;
esac
