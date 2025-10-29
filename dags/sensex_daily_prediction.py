"""
Apache Airflow DAG for model inference and daily predictions
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'catchup': False
}

# Create DAG
dag = DAG(
    'sensex_daily_prediction',
    default_args=default_args,
    description='Daily SENSEX prediction pipeline',
    schedule_interval='0 18 * * 1-5',  # Run at 6 PM on weekdays (after market close)
    max_active_runs=1,
    tags=['sensex', 'prediction', 'inference']
)

def fetch_latest_data(**context):
    """
    Fetch latest data for prediction
    """
    try:
        from src.data.fetch_data import SensexDataFetcher
        from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX, PATHS
        
        # Fetch only latest 60 days of data for prediction
        fetcher = SensexDataFetcher(SENSEX_30_SYMBOLS, SENSEX_INDEX)
        latest_data = fetcher.fetch_all_stocks_data(period="60d", interval="1d")
        
        # Save to a separate prediction data directory
        prediction_data_dir = os.path.join(PATHS['raw_data'], 'prediction')
        fetcher.save_raw_data(latest_data, prediction_data_dir)
        
        return "Latest data fetched successfully"
        
    except Exception as e:
        raise Exception(f"Latest data fetching failed: {str(e)}")

def create_prediction_features(**context):
    """
    Create features for prediction
    """
    try:
        from src.data.create_feature_maps import FeatureEngineer
        from src.data.fetch_data import SensexDataFetcher
        from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX, PATHS
        
        # Load latest data
        fetcher = SensexDataFetcher(SENSEX_30_SYMBOLS, SENSEX_INDEX)
        prediction_data_dir = os.path.join(PATHS['raw_data'], 'prediction')
        stock_data = fetcher.load_raw_data(prediction_data_dir)
        
        # Create features
        engineer = FeatureEngineer(SENSEX_30_SYMBOLS)
        feature_maps, targets, dates = engineer.create_component_feature_maps(
            stock_data, SENSEX_INDEX
        )
        
        # Save prediction features
        prediction_features_dir = os.path.join(PATHS['processed_data'], 'prediction')
        engineer.save_feature_maps(feature_maps, targets, dates, prediction_features_dir)
        
        return "Prediction features created successfully"
        
    except Exception as e:
        raise Exception(f"Prediction feature creation failed: {str(e)}")

def make_prediction(**context):
    """
    Make SENSEX direction prediction
    """
    try:
        from src.utils.mlflow_utils import MLflowManager
        from src.models.convlstm_model import SensexConvLSTM
        from configs.config import MODEL_CONFIG, PATHS
        import numpy as np
        import json
        
        # Load prediction data
        prediction_features_dir = os.path.join(PATHS['processed_data'], 'prediction')
        feature_maps = np.load(os.path.join(prediction_features_dir, 'feature_maps.npy'))
        
        # Initialize MLflow manager
        mlflow_manager = MLflowManager()
        
        # Load the production model
        try:
            model = mlflow_manager.load_model("models:/sensex_convlstm/Production")
        except:
            # Fallback to staging model
            model = mlflow_manager.load_model("models:/sensex_convlstm/Staging")
        
        # Prepare data for prediction (take last 30 days)
        input_shape = (MODEL_CONFIG['sequence_length'], 
                      MODEL_CONFIG['n_stocks'], 
                      MODEL_CONFIG['n_features'])
        convlstm_model = SensexConvLSTM(input_shape, MODEL_CONFIG)
        
        if len(feature_maps) >= MODEL_CONFIG['sequence_length']:
            # Take the last sequence for prediction
            last_sequence = feature_maps[-MODEL_CONFIG['sequence_length']:]
            X_pred = last_sequence.reshape(1, *last_sequence.shape, 1)
            
            # Make prediction
            prob, pred = convlstm_model.predict(X_pred)
            
            # Create prediction result
            prediction_result = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'prediction': int(pred[0][0]),
                'probability': float(prob[0][0]),
                'direction': 'UP' if pred[0][0] == 1 else 'DOWN',
                'confidence': 'HIGH' if abs(prob[0][0] - 0.5) > 0.3 else 'MEDIUM' if abs(prob[0][0] - 0.5) > 0.1 else 'LOW'
            }
            
            # Save prediction
            prediction_file = os.path.join(PATHS['processed_data'], 'latest_prediction.json')
            with open(prediction_file, 'w') as f:
                json.dump(prediction_result, f, indent=2)
            
            return f"Prediction made: {prediction_result['direction']} with {prediction_result['confidence']} confidence"
        else:
            return "Insufficient data for prediction"
            
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def log_prediction_metrics(**context):
    """
    Log prediction metrics to MLflow
    """
    try:
        from src.utils.mlflow_utils import MLflowManager
        import json
        
        # Load prediction result
        prediction_file = os.path.join('/opt/airflow/data/processed', 'latest_prediction.json')
        
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as f:
                prediction_result = json.load(f)
            
            # Initialize MLflow manager
            mlflow_manager = MLflowManager()
            
            # Log prediction metrics
            with mlflow_manager.start_run(run_name="daily_prediction"):
                mlflow_manager.log_metrics({
                    'prediction_probability': prediction_result['probability'],
                    'prediction_binary': prediction_result['prediction']
                })
                
                # Log prediction as parameter
                mlflow_manager.log_params({
                    'prediction_date': prediction_result['date'],
                    'predicted_direction': prediction_result['direction'],
                    'confidence_level': prediction_result['confidence']
                })
            
            return "Prediction logged to MLflow"
        else:
            return "No prediction file found to log"
            
    except Exception as e:
        raise Exception(f"Prediction logging failed: {str(e)}")

# Define tasks
fetch_latest_task = PythonOperator(
    task_id='fetch_latest_data',
    python_callable=fetch_latest_data,
    dag=dag
)

create_pred_features_task = PythonOperator(
    task_id='create_prediction_features',
    python_callable=create_prediction_features,
    dag=dag
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag
)

log_metrics_task = PythonOperator(
    task_id='log_prediction_metrics',
    python_callable=log_prediction_metrics,
    dag=dag
)

# Define task dependencies
fetch_latest_task >> create_pred_features_task >> make_prediction_task >> log_metrics_task
