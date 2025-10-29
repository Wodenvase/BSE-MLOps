"""
MLflow Model Registry Management
Handles model promotion, versioning, and production serving
"""

import mlflow
import mlflow.keras
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    MLflow Model Registry Manager for SENSEX ConvLSTM models
    Handles model promotion, versioning, and production deployment
    """
    
    def __init__(self, tracking_uri: str = None):
        """
        Initialize Model Registry
        
        Args:
            tracking_uri: MLflow tracking server URI (defaults to local)
        """
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local MLflow directory
            mlflow.set_tracking_uri("file:./mlruns")
        
        self.client = MlflowClient()
        self.model_name = "sensex-convlstm-model"
        
        logger.info(f"Initialized Model Registry with tracking URI: {mlflow.get_tracking_uri()}")
    
    def create_registered_model(self) -> bool:
        """
        Create registered model if it doesn't exist
        
        Returns:
            bool: True if successful
        """
        try:
            # Check if model already exists
            try:
                model = self.client.get_registered_model(self.model_name)
                logger.info(f"Model '{self.model_name}' already exists")
                return True
            except:
                # Create new registered model
                model = self.client.create_registered_model(
                    name=self.model_name,
                    description="ConvLSTM model for SENSEX next-day movement prediction"
                )
                logger.info(f"Created registered model: {self.model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating registered model: {str(e)}")
            return False
    
    def get_best_model_from_experiments(self, metric_name: str = "test_accuracy") -> Optional[Dict]:
        """
        Find the best model from all experiments based on a metric
        
        Args:
            metric_name: Metric to optimize for
            
        Returns:
            Dict: Best model information or None
        """
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            
            best_run = None
            best_metric = -1
            
            for experiment in experiments:
                if experiment.name != "Default":  # Skip default experiment
                    # Get runs from this experiment
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="",
                        run_view_type=ViewType.ACTIVE_ONLY,
                        max_results=1000
                    )
                    
                    for run in runs:
                        if run.data.metrics.get(metric_name):
                            metric_value = run.data.metrics[metric_name]
                            if metric_value > best_metric:
                                best_metric = metric_value
                                best_run = run
            
            if best_run:
                logger.info(f"Found best model: Run ID {best_run.info.run_id} with {metric_name}={best_metric:.4f}")
                return {
                    'run_id': best_run.info.run_id,
                    'experiment_id': best_run.info.experiment_id,
                    'metrics': best_run.data.metrics,
                    'params': best_run.data.params,
                    'artifact_uri': best_run.info.artifact_uri
                }
            else:
                logger.warning("No models found with the specified metric")
                return None
                
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            return None
    
    def register_model_version(self, run_id: str, description: str = None) -> Optional[str]:
        """
        Register a new model version from a run
        
        Args:
            run_id: MLflow run ID containing the model
            description: Version description
            
        Returns:
            str: Model version number or None
        """
        try:
            # Ensure registered model exists
            self.create_registered_model()
            
            # Register model version
            model_uri = f"runs:/{run_id}/model"
            
            if not description:
                # Get run info for description
                run = self.client.get_run(run_id)
                test_acc = run.data.metrics.get('test_accuracy', 0)
                train_acc = run.data.metrics.get('train_accuracy', 0)
                description = f"Model from run {run_id[:8]} - Test Acc: {test_acc:.3f}, Train Acc: {train_acc:.3f}"
            
            model_version = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                description=description
            )
            
            logger.info(f"Registered model version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            return None
    
    def promote_model_to_production(self, version: str = None, 
                                  min_accuracy: float = 0.55) -> bool:
        """
        Promote a model version to Production stage
        
        Args:
            version: Specific version to promote (None for latest)
            min_accuracy: Minimum accuracy threshold for promotion
            
        Returns:
            bool: True if successful
        """
        try:
            # Get latest version if not specified
            if not version:
                versions = self.client.get_latest_versions(
                    self.model_name, 
                    stages=["None", "Staging"]
                )
                if not versions:
                    logger.error("No model versions found")
                    return False
                
                # Find best version based on accuracy
                best_version = None
                best_accuracy = 0
                
                for v in versions:
                    # Get run metrics
                    run = self.client.get_run(v.run_id)
                    test_acc = run.data.metrics.get('test_accuracy', 0)
                    
                    if test_acc > best_accuracy and test_acc >= min_accuracy:
                        best_accuracy = test_acc
                        best_version = v
                
                if not best_version:
                    logger.error(f"No model versions meet minimum accuracy threshold: {min_accuracy}")
                    return False
                
                version = best_version.version
                logger.info(f"Selected version {version} with accuracy {best_accuracy:.3f}")
            
            # Archive current production model if exists
            try:
                current_prod = self.client.get_latest_versions(
                    self.model_name, 
                    stages=["Production"]
                )
                
                for prod_model in current_prod:
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=prod_model.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    logger.info(f"Archived previous production version {prod_model.version}")
            except:
                logger.info("No existing production model to archive")
            
            # Promote new version to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                archive_existing_versions=False
            )
            
            logger.info(f"Promoted version {version} to Production stage")
            
            # Log promotion event
            self._log_promotion_event(version)
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model to production: {str(e)}")
            return False
    
    def get_production_model(self) -> Optional[Dict]:
        """
        Get the current production model information
        
        Returns:
            Dict: Production model info or None
        """
        try:
            production_models = self.client.get_latest_versions(
                self.model_name,
                stages=["Production"]
            )
            
            if not production_models:
                logger.warning("No production model found")
                return None
            
            prod_model = production_models[0]  # Should be only one
            
            # Get run information
            run = self.client.get_run(prod_model.run_id)
            
            return {
                'version': prod_model.version,
                'run_id': prod_model.run_id,
                'model_uri': prod_model.source,
                'stage': prod_model.current_stage,
                'creation_timestamp': prod_model.creation_timestamp,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'description': prod_model.description
            }
            
        except Exception as e:
            logger.error(f"Error getting production model: {str(e)}")
            return None
    
    def load_production_model(self):
        """
        Load the production model for inference
        
        Returns:
            Loaded model or None
        """
        try:
            prod_info = self.get_production_model()
            if not prod_info:
                logger.error("No production model available")
                return None
            
            model_uri = f"models:/{self.model_name}/Production"
            model = mlflow.keras.load_model(model_uri)
            
            logger.info(f"Loaded production model version {prod_info['version']}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            return None
    
    def list_all_versions(self) -> List[Dict]:
        """
        List all model versions with their information
        
        Returns:
            List of model version information
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            version_info = []
            for version in versions:
                # Get run metrics
                try:
                    run = self.client.get_run(version.run_id)
                    metrics = run.data.metrics
                except:
                    metrics = {}
                
                version_info.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_timestamp': version.creation_timestamp,
                    'description': version.description,
                    'metrics': metrics
                })
            
            # Sort by version number
            version_info.sort(key=lambda x: int(x['version']), reverse=True)
            
            return version_info
            
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            return []
    
    def evaluate_model_drift(self, current_metrics: Dict, 
                           threshold: float = 0.05) -> Dict:
        """
        Evaluate if current production model shows drift
        
        Args:
            current_metrics: Latest model performance metrics
            threshold: Performance degradation threshold
            
        Returns:
            Dict: Drift analysis results
        """
        try:
            prod_model = self.get_production_model()
            if not prod_model:
                return {'has_drift': False, 'reason': 'No production model'}
            
            prod_metrics = prod_model['metrics']
            
            drift_analysis = {
                'has_drift': False,
                'degraded_metrics': [],
                'improvements': [],
                'production_version': prod_model['version'],
                'comparison': {}
            }
            
            # Compare key metrics
            key_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_auc']
            
            for metric in key_metrics:
                if metric in prod_metrics and metric in current_metrics:
                    prod_value = prod_metrics[metric]
                    current_value = current_metrics[metric]
                    
                    degradation = prod_value - current_value
                    drift_analysis['comparison'][metric] = {
                        'production': prod_value,
                        'current': current_value,
                        'degradation': degradation
                    }
                    
                    if degradation > threshold:
                        drift_analysis['has_drift'] = True
                        drift_analysis['degraded_metrics'].append(metric)
                    elif degradation < -threshold:
                        drift_analysis['improvements'].append(metric)
            
            return drift_analysis
            
        except Exception as e:
            logger.error(f"Error evaluating model drift: {str(e)}")
            return {'has_drift': False, 'error': str(e)}
    
    def _log_promotion_event(self, version: str):
        """Log model promotion event"""
        try:
            event_data = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'version': version,
                'stage': 'Production',
                'event_type': 'promotion'
            }
            
            # Save to local log file
            log_file = 'model_promotion_log.json'
            events = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            
            events.append(event_data)
            
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging promotion event: {str(e)}")
    
    def setup_automated_promotion(self) -> bool:
        """
        Setup automated model promotion workflow
        
        Returns:
            bool: True if successful
        """
        try:
            # Find and register best model
            best_model = self.get_best_model_from_experiments()
            
            if not best_model:
                logger.error("No suitable model found for promotion")
                return False
            
            # Register model version
            version = self.register_model_version(
                best_model['run_id'],
                f"Auto-registered best model - Accuracy: {best_model['metrics'].get('test_accuracy', 0):.3f}"
            )
            
            if not version:
                logger.error("Failed to register model version")
                return False
            
            # Promote to production
            success = self.promote_model_to_production(version)
            
            if success:
                logger.info("Automated model promotion completed successfully")
                return True
            else:
                logger.error("Failed to promote model to production")
                return False
                
        except Exception as e:
            logger.error(f"Error in automated promotion: {str(e)}")
            return False

def main():
    """
    Main function for testing Model Registry
    """
    print("🚀 Setting up MLflow Model Registry...")
    
    registry = ModelRegistry()
    
    # Create registered model
    print("📝 Creating registered model...")
    registry.create_registered_model()
    
    # Setup automated promotion
    print("🔄 Setting up automated promotion...")
    success = registry.setup_automated_promotion()
    
    if success:
        print("✅ Model Registry setup completed successfully!")
        
        # Display production model info
        prod_model = registry.get_production_model()
        if prod_model:
            print(f"\n📊 Production Model Info:")
            print(f"Version: {prod_model['version']}")
            print(f"Run ID: {prod_model['run_id']}")
            print(f"Test Accuracy: {prod_model['metrics'].get('test_accuracy', 'N/A')}")
            print(f"Created: {datetime.fromtimestamp(prod_model['creation_timestamp']/1000)}")
        
        # List all versions
        print(f"\n📋 All Model Versions:")
        versions = registry.list_all_versions()
        for v in versions:
            print(f"  Version {v['version']}: {v['stage']} - Accuracy: {v['metrics'].get('test_accuracy', 'N/A')}")
    
    else:
        print("❌ Model Registry setup failed")

if __name__ == "__main__":
    main()
