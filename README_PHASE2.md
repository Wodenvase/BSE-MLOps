# SENSEX MLOps Pipeline - Phase 2: Experimentation, Training & Tracking

## 🎯 Overview

Phase 2 focuses on **rigorous model training and experimentation** with comprehensive tracking for reproducibility. We implement ConvLSTM models with advanced hyperparameter optimization, containerized training environments, and extensive experiment tracking using MLflow.

### Key Objectives
- ✅ **Containerized Training**: Isolated training environment with all ML dependencies
- ✅ **Experiment Tracking**: Comprehensive MLflow integration for reproducible experiments
- ✅ **Hyperparameter Optimization**: Automated tuning using Optuna with TPE sampling
- ✅ **DVC Data Loading**: Direct data access from versioned remote storage
- ✅ **Advanced Model Architecture**: Configurable ConvLSTM with regularization
- ✅ **Comprehensive Evaluation**: Detailed model analysis with trading insights

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Training Container │    │   MLflow Tracking   │    │  Optuna Optimization│
│                     │    │                     │    │                     │
│ • TensorFlow 2.13   │────▶ • Experiment Logs   │◀───│ • TPE Sampler       │
│ • Keras ConvLSTM    │    │ • Model Registry    │    │ • MedianPruner      │
│ • DVC Data Loading  │    │ • Artifact Storage  │    │ • 100+ Trials       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Comprehensive Evaluation Pipeline                    │
│ • Confusion Matrix  • ROC/PR Curves  • Trading Analysis  • Time Series     │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Model Artifacts   │    │   Experiment Reports│    │   Production Models │
│                     │    │                     │    │                     │
│ • Trained Models    │    │ • HTML Reports      │    │ • Best Performers   │
│ • Model Metadata    │    │ • Visualizations    │    │ • Ensemble Candidates│
│ • Evaluation Metrics│    │ • Trading Insights  │    │ • Phase 3 Ready     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 Phase 2 Components

### 🐳 **Containerized Training Environment**
```dockerfile
# Dockerfile.training - Optimized ML training container
- TensorFlow 2.13 with GPU support
- MLflow 2.7.1 for experiment tracking
- Optuna 3.4.0 for hyperparameter optimization
- DVC with Google Drive integration
- Comprehensive ML stack (scikit-learn, matplotlib, seaborn)
```

### 🧠 **Advanced ConvLSTM Model**
```python
# src/models/convlstm_model.py
- Configurable ConvLSTM layers with dropout
- Dense layers with batch normalization
- L1/L2 regularization
- Advanced callbacks (EarlyStopping, ReduceLROnPlateau)
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)
```

**Model Architecture:**
- **Input**: (sequence_length, n_stocks, n_features, 1)
- **ConvLSTM Layers**: Configurable filters [64, 32, 16] with (3,3) kernels
- **Global Pooling**: Reduces spatial dimensions
- **Dense Layers**: [128, 64, 32] with dropout and regularization
- **Output**: Sigmoid activation for binary classification

### 🔬 **Training Pipeline** (`train.py`)
```python
# Comprehensive training with MLflow integration
- DVC data loading with fallback to local storage
- Time-based data splitting for time series
- Extensive parameter and metric logging
- Automatic visualization generation
- Model checkpointing and artifact management
```

### 🎯 **Hyperparameter Optimization** (`hyperparameter_tuning.py`)
```python
# Optuna-based optimization with MLflow tracking
- TPE (Tree-structured Parzen Estimator) sampler
- MedianPruner for early stopping poor trials
- 100+ hyperparameter combinations tested
- Automatic best model training with extended epochs
```

**Optimized Hyperparameters:**
- ConvLSTM filters: 16-128 per layer
- Kernel sizes: 2x2 to 4x4
- Dropout rates: 0.1-0.4
- Dense units: 32-256 per layer
- Learning rates: 1e-5 to 1e-2 (log scale)
- Batch sizes: 16, 32, 64, 128
- Regularization: L1/L2 from 1e-5 to 1e-2

### 📊 **Model Evaluation** (`model_evaluation.py`)
```python
# Comprehensive evaluation with trading analysis
- Standard ML metrics (Accuracy, Precision, Recall, F1, AUC)
- Trading-specific metrics (Up/Down day accuracy)
- Confidence-based analysis
- Time series performance visualization
- HTML report generation with insights
```

### 🚀 **Pipeline Orchestration** (`train_pipeline.sh`)
```bash
# Complete training pipeline automation
- Environment setup and validation
- Docker image building and management
- Experiment suite execution
- Monitoring and cleanup utilities
```

## 🛠️ Usage Guide

### Prerequisites
- Phase 1 completed with processed data in `data/processed/`
- Docker and Docker Compose installed
- MLflow server running (handled automatically)
- Sufficient compute resources (4GB+ RAM, GPU optional)

### 🚀 **Quick Start**

#### 1. **Complete Training Suite** (Recommended)
```bash
# Run comprehensive experiment suite
./train_pipeline.sh suite

# This includes:
# - Baseline model training
# - Quick hyperparameter search (20 trials)
# - Extended optimization (100 trials)
# - Best model evaluation
```

#### 2. **Individual Commands**
```bash
# Setup environment
./train_pipeline.sh setup

# Build training container
./train_pipeline.sh build

# Basic model training
./train_pipeline.sh train

# Hyperparameter tuning
./train_pipeline.sh tune 50        # 50 trials
./train_pipeline.sh tune 100 3600  # 100 trials, 1-hour timeout

# Evaluate latest model
./train_pipeline.sh evaluate

# Check status and monitoring
./train_pipeline.sh status
```

#### 3. **Direct Container Usage**
```bash
# Build training image
docker build -f Dockerfile.training -t sensex-training:latest .

# Run training with custom config
docker run --rm \
  --network sensex-mlops_default \
  -v $(pwd):/app \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  sensex-training:latest \
  python3 train.py --run-name "custom_experiment"

# Run hyperparameter tuning
docker run --rm \
  --network sensex-mlops_default \
  -v $(pwd):/app \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  sensex-training:latest \
  python3 hyperparameter_tuning.py --n-trials 50 --train-best
```

### 📊 **Accessing Results**

#### MLflow UI
```bash
# Access experiment tracking
open http://localhost:5000

# Compare experiments, view metrics, download models
# Navigate to Experiments → sensex-convlstm-phase2
```

#### Experiment Artifacts
```bash
# View experiment results
ls experiments/

# Latest experiment suite
ls experiments/suite_*/

# Model files
ls models/trained/

# Evaluation reports
ls experiments/evaluation_*/
```

## 📈 **Configuration Files**

### **Training Configuration** (`configs/training_config.json`)
```json
{
  "experiment_name": "sensex-convlstm-phase2",
  "model_params": {
    "sequence_length": 30,
    "conv_lstm_filters": [64, 32, 16],
    "dense_units": [128, 64, 32],
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15
  },
  "data_split": {
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1
  }
}
```

### **Hyperparameter Tuning Configuration** (`configs/tuning_config.json`)
```json
{
  "optimization": {
    "n_trials": 100,
    "objective_metric": "val_f1_score",
    "direction": "maximize"
  },
  "hyperparameter_space": {
    "conv_lstm_filters": {"min": 16, "max": 128},
    "learning_rate": [1e-5, 1e-2],
    "batch_size": [16, 32, 64, 128]
  }
}
```

## 📊 **Experiment Tracking**

### MLflow Integration
- **Experiments**: Organized by model type and optimization method
- **Parameters**: All hyperparameters logged automatically
- **Metrics**: Training/validation metrics tracked per epoch
- **Artifacts**: Models, plots, and evaluation reports
- **Model Registry**: Best models registered for production use

### Key Metrics Tracked
```python
# Training Metrics
- train_loss, train_accuracy, train_f1_score
- val_loss, val_accuracy, val_f1_score
- training_time_seconds

# Test Metrics  
- test_accuracy, test_precision, test_recall, test_f1_score
- test_auc, roc_auc_score
- up_day_accuracy, down_day_accuracy

# Model Info
- model_parameters, input_shape
- best_epoch, total_epochs
```

### Optuna Study Tracking
```python
# Optimization Metrics
- trial_number, trial_value (F1 score)
- parameter_importance, optimization_history
- best_trial_params, convergence_analysis

# Study Artifacts
- study_summary.json, all_trials.json
- trials_dataframe.csv, optimization_visualizations
```

## 📊 **Model Evaluation Reports**

### Comprehensive Analysis
- **Performance Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Trading Analysis**: Up/down day accuracy, confidence-based performance
- **Visual Analysis**: Confusion matrix, ROC/PR curves, prediction distribution
- **Time Series Analysis**: Performance over time, rolling accuracy
- **HTML Reports**: Detailed insights with recommendations

### Sample Evaluation Output
```
📊 Key Results:
  Accuracy: 0.6234
  F1-Score: 0.6451
  ROC-AUC: 0.6789
  Up Day Accuracy: 0.6112
  Down Day Accuracy: 0.6356
  High Confidence Accuracy: 0.7234
```

## 🔍 **Monitoring and Debugging**

### Real-time Monitoring
```bash
# MLflow experiments
open http://localhost:5000

# Training logs
tail -f logs/training.log

# Hyperparameter tuning progress
tail -f logs/hyperparameter_tuning.log

# Docker container logs
docker logs -f <container_name>
```

### Common Issues and Solutions

#### **GPU Configuration**
```bash
# Check GPU availability
docker run --rm sensex-training:latest python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Enable GPU support (if available)
# Uncomment GPU sections in docker-compose.yml
```

#### **Memory Issues**
```bash
# Reduce batch size in configs/training_config.json
"batch_size": 16  # Instead of 32

# Limit concurrent trials in tuning
./train_pipeline.sh tune 20  # Instead of 100
```

#### **DVC Data Loading Issues**
```bash
# Ensure DVC is configured
dvc remote list

# Check data availability
ls data/processed/

# Manual data pull
dvc pull
```

## 🎯 **Expected Results**

### Performance Targets
- **Baseline Accuracy**: 55-60% (better than random)
- **Optimized F1-Score**: 60-65% (good trading signal)
- **ROC-AUC**: 65-70% (strong discriminative ability)
- **High Confidence Accuracy**: 70%+ (reliable predictions)

### Training Characteristics
- **Convergence**: 30-80 epochs typically
- **Training Time**: 5-15 minutes per model (CPU), 2-5 minutes (GPU)
- **Hyperparameter Trials**: 50-100 for good optimization
- **Best Models**: Top 5-10% of trials show significant improvement

## 🚀 **Next Steps: Phase 3 Preview**

Phase 2 prepares for production deployment with:

1. **Model Selection**: Best performing models identified and validated
2. **Performance Analysis**: Trading implications thoroughly analyzed
3. **Reproducibility**: All experiments tracked and reproducible
4. **Production Candidates**: Models ready for real-time inference

Phase 3 will focus on:
- Real-time inference API development
- Model serving with version control
- A/B testing framework
- Production monitoring and alerting
- Continuous integration/deployment

## 📂 **File Structure Summary**

```
sensex-mlops/
├── 🔬 Phase 2: Training & Experimentation
│   ├── Dockerfile.training              # ML training environment
│   ├── train.py                         # Comprehensive training script
│   ├── hyperparameter_tuning.py         # Optuna optimization
│   ├── model_evaluation.py              # Detailed model evaluation
│   ├── train_pipeline.sh                # Training orchestration
│   ├── configs/
│   │   ├── training_config.json         # Training parameters
│   │   └── tuning_config.json          # Optimization settings
│   ├── src/models/
│   │   └── convlstm_model.py           # Enhanced ConvLSTM architecture
│   ├── experiments/                     # Experiment artifacts
│   ├── models/
│   │   ├── trained/                     # Final trained models
│   │   ├── artifacts/                   # Model metadata
│   │   └── optuna_trials/              # Trial models
│   └── logs/                           # Training logs
├── 📊 MLflow Integration
│   └── mlruns/                         # MLflow tracking data
└── 🐳 Docker Enhancement
    └── docker-compose.yml              # Updated with training service
```

---

**📊 Phase 2 Status: COMPLETED ✅**

- ✅ Containerized training environment with full ML stack
- ✅ Advanced ConvLSTM architecture with configurable hyperparameters  
- ✅ Comprehensive training pipeline with MLflow integration
- ✅ Automated hyperparameter optimization using Optuna
- ✅ Detailed model evaluation with trading analysis
- ✅ Complete pipeline orchestration and monitoring tools

**🚀 Ready for Phase 3: Production Deployment & Real-time Inference**
