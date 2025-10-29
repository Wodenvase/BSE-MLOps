"""
Production Airflow DAG for SENSEX Data Engineering Pipeline - Phase 1
Orchestrates: ticker scraping -> data fetching -> feature processing -> DVC versioning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, '/opt/airflow/src')

# Default arguments
default_args = {
    'owner': 'sensex-mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
    'max_active_runs': 1,
}

# DAG definition
dag = DAG(
    'sensex_data_engineering_pipeline_v1',
    default_args=default_args,
    description='Phase 1: Automated Data Engineering Pipeline for SENSEX Components',
    schedule_interval='0 6 * * 1-5',  # 6 AM on weekdays
    max_active_runs=1,
    tags=['phase-1', 'data-engineering', 'sensex', 'production']
)

# ============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ============================================================================

def get_pipeline_config():
    """Get pipeline configuration"""
    return {
        'data_period': '2y',
        'min_stocks_required': 25,
        'target_features': 40,
        'feature_selection_method': 'variance',
        'parallel_fetching': True,
        'max_workers': 8,
        'data_quality_threshold': 0.8,
        'paths': {
            'raw_data': '/opt/airflow/data/raw',
            'processed_data': '/opt/airflow/data/processed',
            'logs': '/opt/airflow/logs',
            'temp': '/opt/airflow/temp'
        }
    }

def setup_directories(**context):
    """Setup required directories"""
    config = get_pipeline_config()
    
    for path_name, path in config['paths'].items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {path}")
    
    return "Directories setup completed"

def check_pipeline_health(**context):
    """Check if pipeline prerequisites are met"""
    config = get_pipeline_config()
    health_checks = []
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('/opt/airflow')
    free_gb = free // (1024**3)
    
    if free_gb < 5:  # Less than 5GB free
        health_checks.append(f"❌ Low disk space: {free_gb}GB free")
    else:
        health_checks.append(f"✅ Disk space: {free_gb}GB free")
    
    # Check required directories
    for path_name, path in config['paths'].items():
        if Path(path).exists():
            health_checks.append(f"✅ Directory exists: {path}")
        else:
            health_checks.append(f"❌ Missing directory: {path}")
    
    # Log health status
    logger.info("PIPELINE HEALTH CHECK:")
    for check in health_checks:
        logger.info(check)
    
    # Return status
    failures = [check for check in health_checks if check.startswith('❌')]
    if failures:
        raise Exception(f"Health check failed: {failures}")
    
    return "Pipeline health check passed"

# ============================================================================
# TASK 1: SENSEX TICKER SCRAPING
# ============================================================================

def scrape_sensex_tickers(**context):
    """Scrape current SENSEX 30 component tickers"""
    logger.info("Starting SENSEX ticker scraping...")
    
    try:
        from data.get_sensex_tickers import SensexTickerScraper
        
        scraper = SensexTickerScraper()
        
        # Try to load from cache first
        tickers = scraper.load_tickers_from_file('data/sensex_components.json')
        
        # If no cache or cache is old, scrape fresh
        if not tickers:
            logger.info("No cached tickers found, scraping fresh...")
            tickers = scraper.get_sensex_components(validate=True)
            scraper.save_tickers_to_file(tickers, 'data/sensex_components.json')
        
        # Validate ticker count
        config = get_pipeline_config()
        if len(tickers) < config['min_stocks_required']:
            raise Exception(f"Insufficient tickers: {len(tickers)} < {config['min_stocks_required']}")
        
        # Store in XCom for next tasks
        context['task_instance'].xcom_push(key='sensex_tickers', value=tickers)
        context['task_instance'].xcom_push(key='ticker_count', value=len(tickers))
        
        logger.info(f"✅ Successfully scraped {len(tickers)} SENSEX tickers")
        return f"Scraped {len(tickers)} tickers successfully"
        
    except Exception as e:
        logger.error(f"❌ Ticker scraping failed: {str(e)}")
        raise

def validate_tickers(**context):
    """Validate scraped tickers"""
    tickers = context['task_instance'].xcom_pull(task_ids='scrape_tickers', key='sensex_tickers')
    
    if not tickers:
        raise Exception("No tickers received from scraping task")
    
    # Basic validation
    valid_tickers = [t for t in tickers if t.endswith('.NS') and len(t) > 5]
    
    if len(valid_tickers) < 25:
        raise Exception(f"Too few valid tickers: {len(valid_tickers)}")
    
    logger.info(f"✅ Validated {len(valid_tickers)} tickers")
    return "Ticker validation passed"

# ============================================================================
# TASK 2: DATA FETCHING
# ============================================================================

def fetch_stock_data(**context):
    """Fetch historical data for all SENSEX stocks"""
    logger.info("Starting stock data fetching...")
    
    try:
        from data.fetch_data import SensexDataFetcher
        
        # Get tickers from previous task
        tickers = context['task_instance'].xcom_pull(task_ids='scrape_tickers', key='sensex_tickers')
        config = get_pipeline_config()
        
        # Initialize fetcher
        index_symbol = "^BSESN"
        fetcher = SensexDataFetcher(
            tickers, 
            index_symbol, 
            max_workers=config['max_workers']
        )
        
        # Fetch data
        stock_data = fetcher.fetch_all_stocks_data(
            period=config['data_period'],
            interval="1d",
            parallel=config['parallel_fetching']
        )
        
        if not stock_data:
            raise Exception("No stock data fetched")
        
        # Save raw data
        fetcher.save_raw_data(stock_data, config['paths']['raw_data'])
        
        # Generate data quality report
        quality_report = fetcher.get_data_quality_report(stock_data)
        
        # Save quality report
        with open(f"{config['paths']['logs']}/data_quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Check data quality
        success_rate = len(stock_data) / (len(tickers) + 1)  # +1 for index
        if success_rate < config['data_quality_threshold']:
            raise Exception(f"Data quality below threshold: {success_rate:.2%}")
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(key='stocks_fetched', value=len(stock_data))
        context['task_instance'].xcom_push(key='data_quality_score', value=success_rate)
        
        logger.info(f"✅ Successfully fetched data for {len(stock_data)} symbols")
        return f"Fetched data for {len(stock_data)} symbols"
        
    except Exception as e:
        logger.error(f"❌ Data fetching failed: {str(e)}")
        raise

def validate_raw_data(**context):
    """Validate fetched raw data"""
    config = get_pipeline_config()
    raw_data_path = Path(config['paths']['raw_data'])
    
    # Check if data files exist
    data_files = list(raw_data_path.glob('*.csv'))
    
    if len(data_files) < 25:
        raise Exception(f"Insufficient data files: {len(data_files)}")
    
    # Basic validation of file sizes
    total_size = sum(f.stat().st_size for f in data_files)
    if total_size < 1024 * 1024:  # Less than 1MB total
        raise Exception(f"Data files too small: {total_size} bytes")
    
    logger.info(f"✅ Validated {len(data_files)} data files, total size: {total_size/1024/1024:.2f}MB")
    return "Raw data validation passed"

# ============================================================================
# TASK 3: FEATURE PROCESSING
# ============================================================================

def process_features(**context):
    """Process raw data into feature maps"""
    logger.info("Starting feature processing...")
    
    try:
        from data.process_features import AdvancedFeatureProcessor
        from data.fetch_data import SensexDataFetcher
        
        # Get configuration
        config = get_pipeline_config()
        tickers = context['task_instance'].xcom_pull(task_ids='scrape_tickers', key='sensex_tickers')
        
        # Load raw data
        index_symbol = "^BSESN"
        fetcher = SensexDataFetcher(tickers, index_symbol)
        stock_data = fetcher.load_raw_data(config['paths']['raw_data'])
        
        if not stock_data:
            raise Exception("No raw data found to process")
        
        # Initialize feature processor
        processor = AdvancedFeatureProcessor(
            tickers, 
            target_features=config['target_features']
        )
        
        # Create feature maps
        feature_maps, targets, dates, feature_names = processor.create_feature_maps(
            stock_data,
            index_symbol,
            feature_selection_method=config['feature_selection_method']
        )
        
        # Save processed data
        processor.save_feature_maps(
            feature_maps, targets, dates, feature_names,
            output_dir=config['paths']['processed_data']
        )
        
        # Validate results
        if feature_maps.shape[0] < 100:  # Less than 100 days of data
            raise Exception(f"Insufficient processed data: {feature_maps.shape[0]} days")
        
        # Push metrics to XCom
        context['task_instance'].xcom_push(key='feature_map_shape', value=feature_maps.shape)
        context['task_instance'].xcom_push(key='target_distribution', value={
            'up_days': int(np.sum(targets)),
            'down_days': int(len(targets) - np.sum(targets)),
            'up_percentage': float(np.mean(targets))
        })
        
        logger.info(f"✅ Successfully processed features: {feature_maps.shape}")
        return f"Processed feature maps: {feature_maps.shape}"
        
    except Exception as e:
        logger.error(f"❌ Feature processing failed: {str(e)}")
        raise

def validate_processed_data(**context):
    """Validate processed feature maps"""
    config = get_pipeline_config()
    processed_path = Path(config['paths']['processed_data'])
    
    # Check required files
    required_files = [
        'feature_maps.npy',
        'targets.npy', 
        'dates.csv',
        'feature_metadata.json'
    ]
    
    for filename in required_files:
        filepath = processed_path / filename
        if not filepath.exists():
            raise Exception(f"Missing processed file: {filename}")
    
    # Load and validate feature maps
    feature_maps = np.load(processed_path / 'feature_maps.npy')
    targets = np.load(processed_path / 'targets.npy')
    
    # Basic shape validation
    if len(feature_maps.shape) != 3:
        raise Exception(f"Invalid feature map shape: {feature_maps.shape}")
    
    if feature_maps.shape[0] != targets.shape[0]:
        raise Exception("Feature maps and targets size mismatch")
    
    # Check for NaN or infinite values
    if np.isnan(feature_maps).any():
        raise Exception("Feature maps contain NaN values")
    
    if np.isinf(feature_maps).any():
        raise Exception("Feature maps contain infinite values")
    
    logger.info(f"✅ Validated processed data: {feature_maps.shape}")
    return "Processed data validation passed"

# ============================================================================
# TASK 4: DATA VERSIONING WITH DVC
# ============================================================================

def prepare_dvc_commit(**context):
    """Prepare data for DVC versioning"""
    logger.info("Preparing DVC commit...")
    
    config = get_pipeline_config()
    
    # Create commit metadata
    commit_metadata = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_run_id': context['run_id'],
        'data_period': config['data_period'],
        'stocks_processed': context['task_instance'].xcom_pull(task_ids='fetch_data', key='stocks_fetched'),
        'feature_map_shape': context['task_instance'].xcom_pull(task_ids='process_features', key='feature_map_shape'),
        'target_distribution': context['task_instance'].xcom_pull(task_ids='process_features', key='target_distribution'),
        'data_quality_score': context['task_instance'].xcom_pull(task_ids='fetch_data', key='data_quality_score')
    }
    
    # Save metadata
    metadata_path = f"{config['paths']['processed_data']}/pipeline_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(commit_metadata, f, indent=2, default=str)
    
    logger.info("✅ DVC commit preparation completed")
    return "DVC preparation successful"

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Setup tasks
setup_task = PythonOperator(
    task_id='setup_directories',
    python_callable=setup_directories,
    dag=dag
)

health_check_task = PythonOperator(
    task_id='pipeline_health_check',
    python_callable=check_pipeline_health,
    dag=dag
)

# Ticker scraping group
with TaskGroup('ticker_scraping', dag=dag) as ticker_group:
    scrape_task = PythonOperator(
        task_id='scrape_tickers',
        python_callable=scrape_sensex_tickers,
    )
    
    validate_tickers_task = PythonOperator(
        task_id='validate_tickers',
        python_callable=validate_tickers,
    )
    
    scrape_task >> validate_tickers_task

# Data fetching group  
with TaskGroup('data_fetching', dag=dag) as data_group:
    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_stock_data,
    )
    
    validate_raw_task = PythonOperator(
        task_id='validate_raw_data',
        python_callable=validate_raw_data,
    )
    
    fetch_task >> validate_raw_task

# Feature processing group
with TaskGroup('feature_processing', dag=dag) as feature_group:
    process_task = PythonOperator(
        task_id='process_features',
        python_callable=process_features,
    )
    
    validate_processed_task = PythonOperator(
        task_id='validate_processed_data',
        python_callable=validate_processed_data,
    )
    
    process_task >> validate_processed_task

# DVC versioning group
with TaskGroup('data_versioning', dag=dag) as dvc_group:
    prepare_dvc_task = PythonOperator(
        task_id='prepare_dvc_commit',
        python_callable=prepare_dvc_commit,
    )
    
    dvc_add_task = BashOperator(
        task_id='dvc_add_data',
        bash_command="""
        cd /opt/airflow && \
        dvc add data/processed/feature_maps.npy && \
        dvc add data/processed/targets.npy && \
        dvc add data/processed/feature_metadata.json && \
        echo "DVC add completed"
        """,
    )
    
    dvc_push_task = BashOperator(
        task_id='dvc_push_data',
        bash_command="""
        cd /opt/airflow && \
        dvc push && \
        echo "DVC push completed"
        """,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    git_commit_task = BashOperator(
        task_id='git_commit_dvc',
        bash_command="""
        cd /opt/airflow && \
        git add data/processed/*.dvc .gitignore && \
        git commit -m "Update data pipeline - $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit" && \
        echo "Git commit completed"
        """,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    prepare_dvc_task >> dvc_add_task >> dvc_push_task >> git_commit_task

# Success notification
success_task = BashOperator(
    task_id='pipeline_success_notification',
    bash_command="""
    echo "🎉 SENSEX Data Engineering Pipeline completed successfully!"
    echo "Timestamp: $(date)"
    echo "Run ID: {{ run_id }}"
    """,
    trigger_rule=TriggerRule.NONE_FAILED,
    dag=dag
)

# Cleanup task
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command="""
    find /opt/airflow/temp -type f -mtime +1 -delete 2>/dev/null || true
    find /opt/airflow/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

# Main pipeline flow
setup_task >> health_check_task >> ticker_group >> data_group >> feature_group >> dvc_group >> success_task >> cleanup_task
