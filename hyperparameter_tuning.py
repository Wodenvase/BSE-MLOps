#!/usr/bin/env python3
"""
Hyperparameter Optimization for SENSEX ConvLSTM - Phase 2
Uses Optuna for advanced hyperparameter tuning with MLflow integration
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

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
from optuna.integration import MLflowCallback
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras

# DVC for data loading
import dvc.api

# Local imports
sys.path.insert(0, '/app/src')
from models.convlstm_model import ConvLSTMModel, ConvLSTMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/hyperparameter_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SensexHyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for SENSEX ConvLSTM using Optuna
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loaded = False
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Load data once
        self.load_data()
        
        # Setup MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow for experiment tracking"""
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'http://mlflow:5000'))
        
        experiment_name = self.config.get('experiment_name', 'sensex-convlstm-hyperparameter-tuning')
        
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
    
    def load_data(self):
        """Load and prepare data for hyperparameter optimization"""
        logger.info("Loading data for hyperparameter optimization...")
        
        try:
            # Load from DVC
            with dvc.api.open('data/processed/feature_maps.npy', mode='rb') as f:
                feature_maps = np.load(f)
            
            with dvc.api.open('data/processed/targets.npy', mode='rb') as f:
                targets = np.load(f)
            
            logger.info(f"Data loaded: feature_maps {feature_maps.shape}, targets {targets.shape}")
            
        except Exception as e:
            logger.warning(f"Failed to load from DVC: {e}. Trying local storage...")
            
            # Fallback to local
            data_path = Path('/app/data/processed')
            feature_maps = np.load(data_path / 'feature_maps.npy')
            targets = np.load(data_path / 'targets.npy')
        
        # Prepare sequences with default config for data preparation
        temp_config = ConvLSTMConfig()
        temp_model = ConvLSTMModel(temp_config)
        X, y = temp_model.prepare_data(feature_maps, targets)
        
        # Time-based split
        train_size = int(len(X) * self.config['data_split']['train_ratio'])
        val_size = int(len(X) * self.config['data_split']['val_ratio'])
        
        self.X_train = X[:train_size]
        self.X_val = X[train_size:train_size + val_size]
        self.X_test = X[train_size + val_size:]
        
        self.y_train = y[:train_size]
        self.y_val = y[train_size:train_size + val_size]
        self.y_test = y[train_size + val_size:]
        
        self.data_loaded = True
        
        logger.info(f"Data prepared for tuning:")
        logger.info(f"  Training: {self.X_train.shape[0]} samples")
        logger.info(f"  Validation: {self.X_val.shape[0]} samples")
        logger.info(f"  Test: {self.X_test.shape[0]} samples")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        # ConvLSTM architecture parameters
        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
        conv_lstm_filters = []
        
        for i in range(n_conv_layers):
            if i == 0:
                # First layer can be larger
                filters = trial.suggest_int(f'conv_lstm_filters_{i}', 32, 128, step=16)
            else:
                # Subsequent layers typically smaller
                prev_filters = conv_lstm_filters[i-1]
                filters = trial.suggest_int(f'conv_lstm_filters_{i}', 16, prev_filters, step=8)
            conv_lstm_filters.append(filters)
        
        # Kernel size
        kernel_height = trial.suggest_int('kernel_height', 2, 4)
        kernel_width = trial.suggest_int('kernel_width', 2, 4)
        
        # Dropout rates
        conv_lstm_dropout = trial.suggest_float('conv_lstm_dropout', 0.1, 0.4, step=0.05)
        conv_lstm_recurrent_dropout = trial.suggest_float('conv_lstm_recurrent_dropout', 0.1, 0.4, step=0.05)
        
        # Dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 2, 4)
        dense_units = []
        
        for i in range(n_dense_layers):
            if i == 0:
                # First dense layer
                units = trial.suggest_int(f'dense_units_{i}', 64, 256, step=16)
            else:
                # Subsequent layers typically smaller
                prev_units = dense_units[i-1]
                units = trial.suggest_int(f'dense_units_{i}', 32, prev_units, step=16)
            dense_units.append(units)
        
        dense_dropout = trial.suggest_float('dense_dropout', 0.2, 0.5, step=0.05)
        
        # Regularization
        l1_reg = trial.suggest_float('l1_reg', 1e-5, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Fixed parameters for optimization
        sequence_length = 30  # Keep fixed for consistency
        epochs = trial.suggest_int('epochs', 20, 100, step=10)  # Shorter for tuning
        patience = max(5, epochs // 10)  # Adaptive patience
        
        hyperparams = {
            'sequence_length': sequence_length,
            'n_stocks': 30,
            'n_features': 40,
            'conv_lstm_filters': conv_lstm_filters,
            'conv_lstm_kernel_size': [kernel_height, kernel_width],
            'conv_lstm_dropout': conv_lstm_dropout,
            'conv_lstm_recurrent_dropout': conv_lstm_recurrent_dropout,
            'dense_units': dense_units,
            'dense_dropout': dense_dropout,
            'l1_reg': l1_reg,
            'l2_reg': l2_reg,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'patience': patience,
            'output_activation': 'sigmoid'
        }
        
        return hyperparams
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation F1 score (metric to maximize)
        """
        try:
            # Get hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Create model
            config = ConvLSTMConfig(**hyperparams)
            model = ConvLSTMModel(config)
            
            # Build model
            keras_model = model.build_model()
            
            # Create model path for this trial
            trial_id = trial.number
            model_path = f"/app/models/optuna_trials/trial_{trial_id}_model.h5"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Train model
            training_metrics = model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                model_path
            )
            
            # Get validation F1 score as optimization target
            val_f1_score = training_metrics['val_f1_score']
            
            # Log additional metrics for analysis
            trial.set_user_attr('val_accuracy', training_metrics['val_accuracy'])
            trial.set_user_attr('val_precision', training_metrics['val_precision'])
            trial.set_user_attr('val_recall', training_metrics['val_recall'])
            trial.set_user_attr('train_f1_score', training_metrics['train_f1_score'])
            trial.set_user_attr('best_epoch', training_metrics['best_epoch'])
            trial.set_user_attr('total_epochs', training_metrics['total_epochs'])
            trial.set_user_attr('model_params', keras_model.count_params())
            
            # Cleanup model to save memory
            del model
            del keras_model
            keras.backend.clear_session()
            
            # Remove model file to save disk space (keep only best models)
            if os.path.exists(model_path):
                os.remove(model_path)
            
            logger.info(f"Trial {trial_id} completed: F1={val_f1_score:.4f}")
            
            return val_f1_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return a poor score for failed trials
            return 0.0
    
    def run_optimization(self, n_trials: int = 100, timeout: Optional[int] = None) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            
        Returns:
            Optuna study object
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create study
        study_name = f"sensex-convlstm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use MLflow callback for tracking
        mlflc = MLflowCallback(
            tracking_uri=self.config.get('mlflow_uri', 'http://mlflow:5000'),
            metric_name="val_f1_score",
            create_experiment=False,
            mlflow_kwargs={
                "nested": True,
            }
        )
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize F1 score
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,  # Random trials before TPE
                n_ei_candidates=24,   # Expected improvement candidates
                seed=42
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Log best results
        best_trial = study.best_trial
        logger.info(f"🏆 Optimization completed!")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best F1 score: {best_trial.value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Save optimization results
        self.save_optimization_results(study)
        
        return study
    
    def save_optimization_results(self, study: optuna.Study):
        """Save optimization results to files"""
        results_dir = f"/app/experiments/optuna_{study.study_name}"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Save study summary
        study_summary = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'optimization_direction': study.direction.name,
            'datetime_start': study.datetime_start.isoformat() if study.datetime_start else None,
            'datetime_complete': datetime.now().isoformat()
        }
        
        with open(f"{results_dir}/study_summary.json", 'w') as f:
            json.dump(study_summary, f, indent=2)
        
        # Save all trials
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'trial_id': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else None
            }
            trials_data.append(trial_data)
        
        with open(f"{results_dir}/all_trials.json", 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save trials as CSV for analysis
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f"{results_dir}/trials_dataframe.csv", index=False)
        
        # Create optimization visualizations
        self.create_optimization_visualizations(study, results_dir)
        
        logger.info(f"Optimization results saved to: {results_dir}")
    
    def create_optimization_visualizations(self, study: optuna.Study, results_dir: str):
        """Create visualization plots for optimization results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Optimization history
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Optimization History')
            
            # 2. Parameter importance
            plt.subplot(2, 2, 2)
            try:
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.title('Parameter Importance')
            except:
                plt.text(0.5, 0.5, 'Parameter importance\nnot available', ha='center', va='center')
                plt.axis('off')
            
            # 3. Parallel coordinate plot (top 10 trials)
            plt.subplot(2, 2, 3)
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=list(study.best_params.keys())[:5])
                plt.title('Parallel Coordinate Plot')
            except:
                plt.text(0.5, 0.5, 'Parallel coordinate\nplot not available', ha='center', va='center')
                plt.axis('off')
            
            # 4. Trial values distribution
            plt.subplot(2, 2, 4)
            trial_values = [trial.value for trial in study.trials if trial.value is not None]
            plt.hist(trial_values, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('F1 Score')
            plt.ylabel('Number of Trials')
            plt.title('Trial Values Distribution')
            plt.axvline(study.best_value, color='red', linestyle='--', label=f'Best: {study.best_value:.4f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/optimization_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Optimization visualizations created")
            
        except Exception as e:
            logger.warning(f"Could not create optimization visualizations: {e}")
    
    def train_best_model(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Train the best model found during optimization
        
        Args:
            study: Completed Optuna study
            
        Returns:
            Training results dictionary
        """
        logger.info("Training best model with full configuration...")
        
        # Get best hyperparameters
        best_params = study.best_params.copy()
        
        # Extend training for best model
        best_params['epochs'] = self.config.get('final_epochs', 200)
        best_params['patience'] = self.config.get('final_patience', 25)
        
        # Create model with best parameters
        config = ConvLSTMConfig(**best_params)
        model = ConvLSTMModel(config)
        
        # Train with MLflow tracking
        with mlflow.start_run(run_name=f"best_model_{study.study_name}") as run:
            # Log parameters
            mlflow.log_params(best_params)
            mlflow.log_param('optimization_study', study.study_name)
            mlflow.log_param('best_trial_number', study.best_trial.number)
            mlflow.log_metric('optimization_best_f1', study.best_value)
            
            # Build model
            keras_model = model.build_model()
            
            # Model path
            model_path = f"/app/models/trained/best_model_{study.study_name}.h5"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Train model
            training_metrics = model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                model_path
            )
            
            # Evaluate on test set
            test_metrics = model.evaluate(self.X_test, self.y_test)
            
            # Log metrics
            mlflow.log_metrics(training_metrics)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
            
            # Log model
            mlflow.tensorflow.log_model(
                keras_model,
                "model",
                registered_model_name=f"sensex-convlstm-best-{datetime.now().strftime('%Y%m%d')}"
            )
            
            results = {
                'study_name': study.study_name,
                'best_trial': study.best_trial.number,
                'optimization_f1': study.best_value,
                'training_metrics': training_metrics,
                'test_metrics': test_metrics,
                'model_path': model_path,
                'run_id': run.info.run_id
            }
            
            logger.info(f"✅ Best model training completed!")
            logger.info(f"  Optimization F1: {study.best_value:.4f}")
            logger.info(f"  Final Test F1: {test_metrics['test_f1_score']:.4f}")
            logger.info(f"  Model saved: {model_path}")
            
            return results


def load_tuning_config(config_path: str = None) -> Dict[str, Any]:
    """Load hyperparameter tuning configuration"""
    if config_path is None:
        config_path = "/app/configs/tuning_config.json"
    
    # Default configuration
    default_config = {
        "experiment_name": "sensex-convlstm-hyperparameter-tuning",
        "mlflow_uri": "http://mlflow:5000",
        "data_split": {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1
        },
        "optimization": {
            "n_trials": 100,
            "timeout": None,
            "final_epochs": 200,
            "final_patience": 25
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
            logger.info("Using default tuning configuration")
    
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default tuning configuration")
    
    return default_config


def main():
    """Main hyperparameter optimization function"""
    parser = argparse.ArgumentParser(description='SENSEX ConvLSTM Hyperparameter Optimization')
    parser.add_argument('--config', type=str, help='Path to tuning configuration file')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--train-best', action='store_true', help='Train best model after optimization')
    
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
    config = load_tuning_config(args.config)
    config['optimization']['n_trials'] = args.n_trials
    if args.timeout:
        config['optimization']['timeout'] = args.timeout
    
    logger.info("🔍 Starting SENSEX ConvLSTM Hyperparameter Optimization")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize optimizer
        optimizer = SensexHyperparameterOptimizer(config)
        
        # Run optimization
        study = optimizer.run_optimization(
            n_trials=config['optimization']['n_trials'],
            timeout=config['optimization'].get('timeout')
        )
        
        # Train best model if requested
        if args.train_best:
            best_results = optimizer.train_best_model(study)
            logger.info("Best model training completed")
        
        logger.info("🎉 Hyperparameter optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
