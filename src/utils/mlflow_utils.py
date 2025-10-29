"""
MLflow setup and utilities for experiment tracking and model management
"""

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import logging
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """
    Manages MLflow experiment tracking and model registry
    """
    
    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "sensex_forecasting"):
        """
        Initialize MLflow manager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """
        Setup MLflow tracking and experiment
        """
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
            # Initialize client
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server: {str(e)}")
            logger.info("MLflow will use local filesystem tracking")
            
            # Fallback to local tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient()
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters
        """
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for time series metrics
        """
        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def log_model(self, model, artifact_path: str, model_type: str = "tensorflow"):
        """
        Log model to MLflow
        
        Args:
            model: Trained model
            artifact_path: Path where model will be stored
            model_type: Type of model (tensorflow, sklearn, etc.)
        """
        try:
            if model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, artifact_path)
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path)
            else:
                mlflow.log_artifact(model, artifact_path)
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log artifacts to MLflow
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path in MLflow artifact store
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts: {str(e)}")
    
    def register_model(self, model_uri: str, model_name: str, stage: str = "Staging"):
        """
        Register model in MLflow Model Registry
        
        Args:
            model_uri: URI of the model
            model_name: Name for the registered model
            stage: Stage for the model (Staging, Production, etc.)
        """
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version} in {stage}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None
    
    def get_best_model(self, experiment_id: Optional[str] = None, metric: str = "test_accuracy"):
        """
        Get the best model from an experiment based on a metric
        
        Args:
            experiment_id: Experiment ID (uses current if None)
            metric: Metric to optimize for
            
        Returns:
            Best run information
        """
        try:
            if experiment_id is None:
                experiment_id = self.experiment_id
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1
            )
            
            if not runs.empty:
                best_run = runs.iloc[0]
                logger.info(f"Best run found with {metric}: {best_run[f'metrics.{metric}']}")
                return best_run.to_dict()
            else:
                logger.warning("No runs found in experiment")
                return None
                
        except Exception as e:
            logger.error(f"Error getting best model: {str(e)}")
            return None
    
    def load_model(self, model_uri: str, model_type: str = "tensorflow"):
        """
        Load model from MLflow
        
        Args:
            model_uri: URI of the model
            model_type: Type of model
            
        Returns:
            Loaded model
        """
        try:
            if model_type == "tensorflow":
                model = mlflow.tensorflow.load_model(model_uri)
            elif model_type == "sklearn":
                model = mlflow.sklearn.load_model(model_uri)
            else:
                model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded {model_type} model from {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def create_model_config(self, config_path: str):
        """
        Create MLflow configuration file
        
        Args:
            config_path: Path to save configuration
        """
        config = {
            'mlflow': {
                'tracking_uri': self.tracking_uri,
                'experiment_name': self.experiment_name,
                'artifact_location': './mlruns',
                'default_model_name': 'sensex_convlstm',
                'model_registry': {
                    'staging_alias': 'staging',
                    'production_alias': 'production'
                }
            },
            'model_serving': {
                'host': '0.0.0.0',
                'port': 5001,
                'workers': 1
            }
        }
        
        # Create directory if it doesn't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created MLflow configuration at {config_path}")


def setup_mlflow_server():
    """
    Setup and start MLflow tracking server
    """
    import subprocess
    import time
    
    # Create mlruns directory
    os.makedirs('./mlruns', exist_ok=True)
    
    # Start MLflow server
    logger.info("Starting MLflow tracking server...")
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            'mlflow', 'server',
            '--backend-store-uri', 'sqlite:///mlruns.db',
            '--default-artifact-root', './mlruns',
            '--host', '0.0.0.0',
            '--port', '5000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is None:
            logger.info("MLflow server started successfully on http://localhost:5000")
            return server_process
        else:
            logger.error("MLflow server failed to start")
            return None
            
    except Exception as e:
        logger.error(f"Error starting MLflow server: {str(e)}")
        return None


def main():
    """
    Main function to setup MLflow
    """
    # Create MLflow manager
    mlflow_manager = MLflowManager()
    
    # Create configuration file
    config_path = './configs/mlflow_config.yaml'
    mlflow_manager.create_model_config(config_path)
    
    # Start MLflow server (optional - can be done separately)
    logger.info("MLflow setup completed!")
    logger.info("To start MLflow server, run: mlflow server --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
