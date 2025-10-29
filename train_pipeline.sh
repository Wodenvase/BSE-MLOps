#!/bin/bash

# SENSEX MLOps Phase 2: Training & Experimentation Pipeline
# Comprehensive training orchestration with experiment tracking and model management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sensex-mlops-phase2"
TRAINING_IMAGE="sensex-training:latest"
EXPERIMENT_PREFIX="exp"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    SENSEX MLOps Phase 2                     ║"
    echo "║              Training & Experimentation Pipeline            ║"
    echo "║                   Orchestration Script                      ║"
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

print_experiment() {
    echo -e "${PURPLE}[EXPERIMENT]${NC} $1"
}

check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if Phase 1 data exists
    if [ ! -f "data/processed/feature_maps.npy" ]; then
        print_error "Phase 1 data not found. Please run Phase 1 pipeline first."
        echo "Expected files:"
        echo "  - data/processed/feature_maps.npy"
        echo "  - data/processed/targets.npy"
        echo "  - data/processed/dates.csv"
        echo "  - data/processed/feature_metadata.json"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed."
        exit 1
    fi
    
    print_success "All prerequisites met"
}

setup_training_environment() {
    print_step "Setting up training environment..."
    
    # Create necessary directories
    mkdir -p {experiments,logs,models/trained,models/artifacts,models/optuna_trials}
    
    # Set permissions
    chmod -R 755 experiments logs models
    
    # Check if MLflow is running
    if ! curl -s http://localhost:5000/health &> /dev/null; then
        print_info "Starting MLflow server..."
        docker-compose up -d mlflow
        sleep 10
    fi
    
    print_success "Training environment ready"
}

build_training_image() {
    print_step "Building training Docker image..."
    
    print_info "Building optimized training image with all ML dependencies..."
    docker build -f Dockerfile.training -t $TRAINING_IMAGE .
    
    print_success "Training image built: $TRAINING_IMAGE"
}

run_basic_training() {
    print_step "Running basic model training..."
    
    local run_name="${EXPERIMENT_PREFIX}_basic_$(date +%Y%m%d_%H%M%S)"
    
    print_experiment "Starting basic training experiment: $run_name"
    
    docker run --rm \
        --network sensex-mlops_default \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/models":/app/models \
        -v "$(pwd)/experiments":/app/experiments \
        -v "$(pwd)/logs":/app/logs \
        -v "$(pwd)/google-credentials.json":/app/google-credentials.json:ro \
        -e PYTHONPATH=/app/src:/app \
        -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
        -e DVC_CACHE_DIR=/app/.dvc/cache \
        -e GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json \
        --name "training_${run_name}" \
        $TRAINING_IMAGE \
        python3 train.py --run-name "$run_name"
    
    print_success "Basic training completed: $run_name"
}

run_hyperparameter_tuning() {
    print_step "Running hyperparameter optimization..."
    
    local n_trials=${1:-50}  # Default 50 trials
    local timeout=${2:-}     # No timeout by default
    
    print_experiment "Starting hyperparameter tuning with $n_trials trials"
    
    local tuning_args="--n-trials $n_trials"
    if [ -n "$timeout" ]; then
        tuning_args="$tuning_args --timeout $timeout"
    fi
    
    docker run --rm \
        --network sensex-mlops_default \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/models":/app/models \
        -v "$(pwd)/experiments":/app/experiments \
        -v "$(pwd)/logs":/app/logs \
        -v "$(pwd)/google-credentials.json":/app/google-credentials.json:ro \
        -e PYTHONPATH=/app/src:/app \
        -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
        -e DVC_CACHE_DIR=/app/.dvc/cache \
        -e GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json \
        --name "tuning_$(date +%Y%m%d_%H%M%S)" \
        $TRAINING_IMAGE \
        python3 hyperparameter_tuning.py $tuning_args --train-best
    
    print_success "Hyperparameter tuning completed"
}

run_model_evaluation() {
    print_step "Running model evaluation..."
    
    # Find the latest trained model
    local latest_model=$(find models/trained -name "*.h5" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_model" ]; then
        print_error "No trained models found in models/trained/"
        return 1
    fi
    
    print_experiment "Evaluating model: $latest_model"
    
    docker run --rm \
        --network sensex-mlops_default \
        -v "$(pwd)":/app \
        -v "$(pwd)/data":/app/data \
        -v "$(pwd)/models":/app/models \
        -v "$(pwd)/experiments":/app/experiments \
        -v "$(pwd)/logs":/app/logs \
        -v "$(pwd)/google-credentials.json":/app/google-credentials.json:ro \
        -e PYTHONPATH=/app/src:/app \
        -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
        -e DVC_CACHE_DIR=/app/.dvc/cache \
        -e GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json \
        --name "evaluation_$(date +%Y%m%d_%H%M%S)" \
        $TRAINING_IMAGE \
        python3 model_evaluation.py --model-path "$latest_model" --log-mlflow
    
    print_success "Model evaluation completed"
}

run_experiment_suite() {
    print_step "Running comprehensive experiment suite..."
    
    local suite_name="suite_$(date +%Y%m%d_%H%M%S)"
    
    print_experiment "Starting experiment suite: $suite_name"
    
    # Create suite directory
    mkdir -p "experiments/$suite_name"
    
    # 1. Baseline model
    print_info "1/4: Training baseline model..."
    run_basic_training
    
    # 2. Quick hyperparameter search
    print_info "2/4: Running quick hyperparameter search..."
    run_hyperparameter_tuning 20  # 20 trials for quick search
    
    # 3. Extended hyperparameter search
    print_info "3/4: Running extended hyperparameter search..."
    run_hyperparameter_tuning 100  # 100 trials for thorough search
    
    # 4. Evaluate best model
    print_info "4/4: Evaluating best model..."
    run_model_evaluation
    
    # Generate suite summary
    print_info "Generating experiment suite summary..."
    
    cat > "experiments/$suite_name/suite_summary.md" << EOF
# Experiment Suite Summary: $suite_name

**Date:** $(date)
**Duration:** Started at $(date)

## Experiments Conducted

1. **Baseline Training**
   - Standard ConvLSTM with default hyperparameters
   - Purpose: Establish performance baseline

2. **Quick Hyperparameter Search**
   - 20 trials using Optuna TPE sampler
   - Purpose: Rapid exploration of hyperparameter space

3. **Extended Hyperparameter Search**
   - 100 trials using Optuna TPE sampler
   - Purpose: Thorough optimization of hyperparameters

4. **Model Evaluation**
   - Comprehensive evaluation of best model
   - Includes visualizations and trading analysis

## Results

- **MLflow Tracking URI:** http://localhost:5000
- **Experiment Artifacts:** experiments/$suite_name/
- **Model Files:** models/trained/
- **Evaluation Reports:** experiments/evaluation_*/

## Next Steps

1. Review MLflow experiments for best performing models
2. Analyze evaluation reports for trading implications
3. Consider ensemble methods with top models
4. Prepare for production deployment (Phase 3)

EOF
    
    print_success "Experiment suite completed: $suite_name"
}

monitor_training() {
    print_step "Monitoring training progress..."
    
    print_info "Training monitoring options:"
    echo "  • MLflow UI: http://localhost:5000"
    echo "  • Training logs: tail -f logs/training.log"
    echo "  • Tuning logs: tail -f logs/hyperparameter_tuning.log"
    echo "  • Evaluation logs: tail -f logs/model_evaluation.log"
    echo ""
    echo "Real-time monitoring:"
    echo "  docker logs -f <container_name>"
    echo ""
    echo "Available experiments:"
    if [ -d "experiments" ]; then
        ls -la experiments/ | grep "^d" | awk '{print "  " $9}'
    else
        echo "  No experiments found"
    fi
}

cleanup_artifacts() {
    print_step "Cleaning up old artifacts..."
    
    # Remove old experiment artifacts (keep last 10)
    if [ -d "experiments" ]; then
        cd experiments
        ls -t | tail -n +11 | xargs -r rm -rf
        cd ..
        print_info "Cleaned up old experiment artifacts"
    fi
    
    # Remove old model trials (keep last 20)
    if [ -d "models/optuna_trials" ]; then
        cd models/optuna_trials
        ls -t | tail -n +21 | xargs -r rm -f
        cd ../..
        print_info "Cleaned up old trial models"
    fi
    
    # Clean up Docker images
    docker image prune -f > /dev/null 2>&1 || true
    
    print_success "Cleanup completed"
}

show_results_summary() {
    print_step "Results Summary"
    
    echo -e "${GREEN}🎉 Phase 2 Training Pipeline Completed!${NC}"
    echo ""
    echo -e "${BLUE}📊 Available Results:${NC}"
    
    # Count experiments
    local exp_count=0
    if [ -d "experiments" ]; then
        exp_count=$(ls -1 experiments/ | wc -l)
    fi
    
    # Count models
    local model_count=0
    if [ -d "models/trained" ]; then
        model_count=$(find models/trained -name "*.h5" | wc -l)
    fi
    
    echo "  • Experiments: $exp_count"
    echo "  • Trained Models: $model_count"
    echo "  • MLflow Tracking: http://localhost:5000"
    echo ""
    
    echo -e "${YELLOW}📁 Key Directories:${NC}"
    echo "  • Experiments: experiments/"
    echo "  • Models: models/trained/"
    echo "  • Logs: logs/"
    echo "  • Artifacts: models/artifacts/"
    echo ""
    
    echo -e "${PURPLE}🚀 Next Steps:${NC}"
    echo "1. Review MLflow experiments to identify best models"
    echo "2. Analyze evaluation reports for trading performance"
    echo "3. Consider ensemble methods with top performers"
    echo "4. Prepare for Phase 3: Production Deployment"
    echo ""
    
    if [ -d "experiments" ] && [ "$(ls experiments/)" ]; then
        echo -e "${BLUE}📈 Latest Experiments:${NC}"
        ls -lat experiments/ | head -6 | tail -5 | awk '{print "  • " $9 " (" $6 " " $7 " " $8 ")"}'
    fi
}

show_usage() {
    echo "SENSEX MLOps Phase 2: Training & Experimentation Pipeline"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup                    Setup training environment"
    echo "  build                    Build training Docker image"
    echo "  train                    Run basic model training"
    echo "  tune [trials] [timeout]  Run hyperparameter tuning"
    echo "  evaluate                 Evaluate latest trained model"
    echo "  suite                    Run comprehensive experiment suite"
    echo "  monitor                  Show monitoring information"
    echo "  cleanup                  Clean up old artifacts"
    echo "  status                   Show current status"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                 # Setup environment"
    echo "  $0 build                 # Build training image"
    echo "  $0 train                 # Train with default config"
    echo "  $0 tune 50               # Tune with 50 trials"
    echo "  $0 tune 100 3600         # Tune with 100 trials, 1-hour timeout"
    echo "  $0 suite                 # Run full experiment suite"
    echo ""
}

# Main execution
main() {
    print_header
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    case "${1:-}" in
        "setup")
            check_prerequisites
            setup_training_environment
            ;;
        "build")
            build_training_image
            ;;
        "train")
            check_prerequisites
            setup_training_environment
            build_training_image
            run_basic_training
            ;;
        "tune")
            check_prerequisites
            setup_training_environment
            build_training_image
            run_hyperparameter_tuning "${2:-50}" "${3:-}"
            ;;
        "evaluate")
            check_prerequisites
            setup_training_environment
            build_training_image
            run_model_evaluation
            ;;
        "suite")
            check_prerequisites
            setup_training_environment
            build_training_image
            run_experiment_suite
            show_results_summary
            ;;
        "monitor")
            monitor_training
            ;;
        "cleanup")
            cleanup_artifacts
            ;;
        "status")
            monitor_training
            show_results_summary
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            # Default: run full pipeline
            check_prerequisites
            setup_training_environment
            build_training_image
            run_experiment_suite
            show_results_summary
            ;;
    esac
}

# Execute main function
main "$@"
