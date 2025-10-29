#!/usr/bin/env python3
"""
SENSEX MLOps Training Script - Phase 2
Comprehensive training pipeline with MLflow experiment tracking, DVC data loading,
and advanced model training capabilities.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras

# MLflow for experiment tracking
import mlflow
import mlflow.tensorflow
import mlflow.sklearn

# DVC for data loading
import dvc.api

# Local imports
sys.path.insert(0, '/app/src')
from models.convlstm_model import ConvLSTMModel, ConvLSTMConfig, ConvLSTMModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SensexTrainingPipeline:
    """
    Comprehensive training pipeline for SENSEX ConvLSTM model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.data_loaded = False
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Setup MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'http://mlflow:5000'))
        
        experiment_name = self.config.get('experiment_name', 'sensex-convlstm-experiments')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
                
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Using default experiment.")
            mlflow.set_experiment("Default")
    
    def load_data_from_dvc(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data from DVC remote storage using DVC API
        
        Returns:
            Tuple of (feature_maps, targets, dates)
        """
        logger.info("Loading data from DVC remote storage...")
        
        try:
            # Load feature maps
            with dvc.api.open('data/processed/feature_maps.npy', mode='rb') as f:
                feature_maps = np.load(f)
            
            # Load targets
            with dvc.api.open('data/processed/targets.npy', mode='rb') as f:
                targets = np.load(f)
            
            # Load dates
            with dvc.api.open('data/processed/dates.csv') as f:
                dates_df = pd.read_csv(f)
                dates = dates_df['date'].tolist()
            
            # Load metadata
            with dvc.api.open('data/processed/feature_metadata.json') as f:
                metadata = json.load(f)
            
            logger.info(f"✅ Data loaded successfully from DVC:")
            logger.info(f"  Feature maps shape: {feature_maps.shape}")
            logger.info(f"  Targets shape: {targets.shape}")
            logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
            logger.info(f"  Features: {len(metadata.get('feature_names', []))}")
            
            self.data_loaded = True
            return feature_maps, targets, dates
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from DVC: {e}")
            logger.info("Attempting to load from local storage...")
            return self.load_data_local()
    
    def load_data_local(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fallback: Load data from local storage
        
        Returns:
            Tuple of (feature_maps, targets, dates)
        """
        try:
            data_path = Path('/app/data/processed')
            
            feature_maps = np.load(data_path / 'feature_maps.npy')
            targets = np.load(data_path / 'targets.npy')
            
            dates_df = pd.read_csv(data_path / 'dates.csv')
            dates = dates_df['date'].tolist()
            
            logger.info(f"✅ Data loaded from local storage:")
            logger.info(f"  Feature maps shape: {feature_maps.shape}")
            logger.info(f"  Targets shape: {targets.shape}")
            
            self.data_loaded = True
            return feature_maps, targets, dates
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from local storage: {e}")
            raise
    
    def prepare_training_data(self, feature_maps: np.ndarray, targets: np.ndarray) -> None:
        """
        Prepare data for training with proper splits
        
        Args:
            feature_maps: Shape (n_samples, n_stocks, n_features)
            targets: Shape (n_samples,)
        """
        logger.info("Preparing training data...")
        
        # Initialize model to get data preparation method
        temp_config = ConvLSTMConfig(**self.config['model_params'])
        temp_model = ConvLSTMModel(temp_config)
        
        # Prepare sequences
        X, y = temp_model.prepare_data(feature_maps, targets)
        
        # Time-based split (important for time series)
        train_size = int(len(X) * self.config['data_split']['train_ratio'])
        val_size = int(len(X) * self.config['data_split']['val_ratio'])
        
        # Split data chronologically
        self.X_train = X[:train_size]
        self.X_val = X[train_size:train_size + val_size]
        self.X_test = X[train_size + val_size:]
        
        self.y_train = y[:train_size]
        self.y_val = y[train_size:train_size + val_size]
        self.y_test = y[train_size + val_size:]
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training: {self.X_train.shape[0]} samples")
        logger.info(f"  Validation: {self.X_val.shape[0]} samples")
        logger.info(f"  Test: {self.X_test.shape[0]} samples")
        
        # Log class distribution
        logger.info(f"Training class distribution:")
        train_up = np.sum(self.y_train)
        train_down = len(self.y_train) - train_up
        logger.info(f"  Up days: {train_up} ({train_up/len(self.y_train)*100:.1f}%)")
        logger.info(f"  Down days: {train_down} ({train_down/len(self.y_train)*100:.1f}%)")
    
    def create_model(self) -> ConvLSTMModel:
        """Create and compile the ConvLSTM model"""
        logger.info("Creating ConvLSTM model...")
        
        # Create model configuration
        model_config = ConvLSTMConfig(**self.config['model_params'])
        
        # Create model
        self.model = ConvLSTMModel(model_config)
        keras_model = self.model.build_model()
        
        logger.info(f"Model created with {keras_model.count_params():,} parameters")
        
        return self.model
    
    def train_model(self, run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model with comprehensive logging
        
        Args:
            run_name: Name for the MLflow run
            
        Returns:
            Dictionary containing training results
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_data_from_dvc() first.")
        
        if self.model is None:
            self.create_model()
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Log parameters
            mlflow.log_params(self.config['model_params'])
            mlflow.log_params({
                'data_split_train': self.config['data_split']['train_ratio'],
                'data_split_val': self.config['data_split']['val_ratio'],
                'data_split_test': self.config['data_split']['test_ratio'],
                'input_shape': list(self.X_train.shape[1:]),
                'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test)
            })
            
            # Log dataset info
            mlflow.log_metrics({
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'train_up_ratio': float(np.mean(self.y_train)),
                'val_up_ratio': float(np.mean(self.y_val)),
                'test_up_ratio': float(np.mean(self.y_test))
            })
            
            # Create model path
            model_path = f"/app/models/trained/convlstm_run_{run_id}.h5"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Train model
            logger.info("Starting model training...")
            start_time = datetime.now()
            
            training_metrics = self.model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                model_path
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log training metrics
            mlflow.log_metrics(training_metrics)
            mlflow.log_metric('training_time_seconds', training_time)
            
            # Evaluate on test set
            logger.info("Evaluating model on test set...")
            test_metrics = self.model.evaluate(self.X_test, self.y_test)
            
            # Log test metrics
            test_metrics_for_mlflow = {k: v for k, v in test_metrics.items() 
                                     if isinstance(v, (int, float))}
            mlflow.log_metrics(test_metrics_for_mlflow)
            
            # Create and log visualizations
            self.create_and_log_visualizations(run_id, test_metrics)
            
            # Log model
            mlflow.tensorflow.log_model(
                self.model.model, 
                "model",
                registered_model_name=f"sensex-convlstm-{datetime.now().strftime('%Y%m%d')}"
            )
            
            # Log artifacts
            mlflow.log_artifact(model_path, "models")
            
            # Prepare results
            results = {
                'run_id': run_id,
                'training_metrics': training_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'model_path': model_path
            }
            
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"  Run ID: {run_id}")
            logger.info(f"  Test Accuracy: {test_metrics['test_accuracy']:.4f}")
            logger.info(f"  Test F1-Score: {test_metrics['test_f1_score']:.4f}")
            logger.info(f"  Training Time: {training_time:.1f} seconds")
            
            return results
    
    def create_and_log_visualizations(self, run_id: str, test_metrics: Dict) -> None:
        """Create and log visualization artifacts"""
        logger.info("Creating visualizations...")
        
        viz_dir = f"/app/experiments/run_{run_id}/visualizations"
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(test_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = f"{viz_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, "visualizations")
        
        # 2. Training History (if available)
        if hasattr(self.model, 'history') and self.model.history:
            history = self.model.history.history
            
            plt.figure(figsize=(15, 5))
            
            # Loss
            plt.subplot(1, 3, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Accuracy
            plt.subplot(1, 3, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # F1 Score
            plt.subplot(1, 3, 3)
            plt.plot(history['f1_score'], label='Training F1')
            plt.plot(history['val_f1_score'], label='Validation F1')
            plt.title('Model F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            
            history_path = f"{viz_dir}/training_history.png"
            plt.savefig(history_path, dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(history_path, "visualizations")
        
        # 3. Model Architecture
        try:
            model_viz_path = f"{viz_dir}/model_architecture.png"
            keras.utils.plot_model(
                self.model.model, 
                to_file=model_viz_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=300
            )
            mlflow.log_artifact(model_viz_path, "visualizations")
        except Exception as e:
            logger.warning(f"Could not create model architecture plot: {e}")
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def save_experiment_results(self, results: Dict, output_path: str = None) -> None:
        """Save experiment results to file"""
        if output_path is None:
            output_path = f"/app/experiments/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_copy = results.copy()
        if 'test_metrics' in results_copy:
            test_metrics = results_copy['test_metrics'].copy()
            for key, value in test_metrics.items():
                if isinstance(value, np.ndarray):
                    test_metrics[key] = value.tolist()
            results_copy['test_metrics'] = test_metrics
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Experiment results saved to: {output_path}")


def load_training_config(config_path: str = None) -> Dict[str, Any]:
    """Load training configuration"""
    if config_path is None:
        config_path = "/app/configs/training_config.json"
    
    # Default configuration
    default_config = {
        "experiment_name": "sensex-convlstm-phase2",
        "mlflow_uri": "http://mlflow:5000",
        "model_params": {
            "sequence_length": 30,
            "n_stocks": 30,
            "n_features": 40,
            "conv_lstm_filters": [64, 32, 16],
            "conv_lstm_kernel_size": [3, 3],
            "conv_lstm_dropout": 0.2,
            "conv_lstm_recurrent_dropout": 0.2,
            "dense_units": [128, 64, 32],
            "dense_dropout": 0.3,
            "l1_reg": 0.001,
            "l2_reg": 0.001,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 15,
            "output_activation": "sigmoid"
        },
        "data_split": {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1
        }
    }
    
    # Try to load custom config
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Merge configs
            default_config.update(custom_config)
            logger.info(f"Configuration loaded from: {config_path}")
        else:
            logger.info("Using default configuration")
    
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
    
    return default_config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='SENSEX ConvLSTM Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to training configuration file')
    parser.add_argument('--run-name', type=str, help='Name for the MLflow run')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("GPU disabled")
    else:
        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} device(s)")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.info("No GPU available, using CPU")
    
    # Load configuration
    config = load_training_config(args.config)
    
    logger.info("🚀 Starting SENSEX ConvLSTM Training Pipeline")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize training pipeline
        pipeline = SensexTrainingPipeline(config)
        
        # Load data
        feature_maps, targets, dates = pipeline.load_data_from_dvc()
        
        # Prepare training data
        pipeline.prepare_training_data(feature_maps, targets)
        
        # Train model
        results = pipeline.train_model(run_name=args.run_name)
        
        # Save results
        pipeline.save_experiment_results(results)
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
