"""
Apache Airflow DAG for SENSEX data pipeline and model training orchestration
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import sys
import os

# Add src directory to Python path
sys.path.append('/opt/airflow/dags/src')

# Default arguments for the DAG
default_args = {
    'owner': 'sensex-mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create DAG
dag = DAG(
    'sensex_data_pipeline',
    default_args=default_args,
    description='SENSEX data fetching, feature engineering, and model training pipeline',
    schedule_interval='0 6 * * 1-5',  # Run at 6 AM on weekdays
    max_active_runs=1,
    tags=['sensex', 'stock-prediction', 'mlops']
)

def fetch_sensex_data(**context):
    """
    Task to fetch SENSEX component data
    """
    try:
        from src.data.fetch_data import main as fetch_main
        fetch_main()
        return "Data fetching completed successfully"
    except Exception as e:
        raise Exception(f"Data fetching failed: {str(e)}")

def create_feature_maps(**context):
    """
    Task to create component feature maps
    """
    try:
        from src.data.create_feature_maps import main as feature_main
        feature_main()
        return "Feature engineering completed successfully"
    except Exception as e:
        raise Exception(f"Feature engineering failed: {str(e)}")

def train_convlstm_model(**context):
    """
    Task to train ConvLSTM model
    """
    try:
        from src.models.convlstm_model import main as model_main
        model_main()
        return "Model training completed successfully"
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")

def validate_data(**context):
    """
    Task to validate processed data quality
    """
    import numpy as np
    import pandas as pd
    
    try:
        # Check if feature maps exist and have correct shape
        feature_maps = np.load('/opt/airflow/data/processed/feature_maps.npy')
        targets = np.load('/opt/airflow/data/processed/targets.npy')
        
        # Validate shapes
        assert len(feature_maps.shape) == 3, f"Feature maps should be 3D, got shape {feature_maps.shape}"
        assert feature_maps.shape[0] == targets.shape[0], "Feature maps and targets should have same number of samples"
        assert feature_maps.shape[1] == 30, "Should have 30 stocks"
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(feature_maps))
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in feature maps")
        
        return f"Data validation passed: {feature_maps.shape}, {targets.shape}"
        
    except Exception as e:
        raise Exception(f"Data validation failed: {str(e)}")

def update_model_registry(**context):
    """
    Task to update model registry with new model
    """
    try:
        from src.utils.mlflow_utils import MLflowManager
        
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Get the best model from the latest experiment
        best_model = mlflow_manager.get_best_model(metric="test_accuracy")
        
        if best_model:
            # Register the best model
            model_uri = f"runs:/{best_model['run_id']}/model"
            mlflow_manager.register_model(
                model_uri=model_uri,
                model_name="sensex_convlstm",
                stage="Staging"
            )
            return "Model registered successfully"
        else:
            return "No model found to register"
            
    except Exception as e:
        raise Exception(f"Model registry update failed: {str(e)}")

# Define tasks
fetch_data_task = PythonOperator(
    task_id='fetch_sensex_data',
    python_callable=fetch_sensex_data,
    dag=dag
)

create_features_task = PythonOperator(
    task_id='create_feature_maps',
    python_callable=create_feature_maps,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_convlstm_model',
    python_callable=train_convlstm_model,
    dag=dag
)

update_registry_task = PythonOperator(
    task_id='update_model_registry',
    python_callable=update_model_registry,
    dag=dag
)

# Create DVC tracking task
dvc_add_task = BashOperator(
    task_id='dvc_track_data',
    bash_command="""
    cd /opt/airflow && \
    dvc add data/processed/feature_maps.npy && \
    dvc add data/processed/targets.npy && \
    dvc add models/ || true
    """,
    dag=dag
)

# Cleanup task
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command="""
    find /opt/airflow/data/raw -name "*.tmp" -delete || true && \
    find /opt/airflow/logs -name "*.log" -mtime +7 -delete || true
    """,
    dag=dag
)

# Define task dependencies
fetch_data_task >> create_features_task >> validate_data_task >> train_model_task >> update_registry_task >> dvc_add_task >> cleanup_task
