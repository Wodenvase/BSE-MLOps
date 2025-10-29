# Test Configuration and Fixtures
import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import tempfile
import shutil

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(110, 120, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(95, 115, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure high >= low and realistic price relationships
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

@pytest.fixture
def sample_sensex_symbols():
    """Sample SENSEX symbols for testing"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS'
    ]

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_mlflow_run():
    """Mock MLflow run data"""
    return {
        'run_id': 'test_run_123',
        'experiment_id': '1',
        'status': 'FINISHED',
        'metrics': {
            'test_accuracy': 0.65,
            'test_precision': 0.62,
            'test_recall': 0.58,
            'test_auc': 0.67
        },
        'params': {
            'sequence_length': 30,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }

@pytest.fixture
def sample_feature_data():
    """Sample processed feature data"""
    np.random.seed(42)
    
    # Shape: (batch_size, sequence_length, n_stocks, n_features)
    batch_size = 10
    sequence_length = 30
    n_stocks = 30
    n_features = 45
    
    return np.random.random((batch_size, sequence_length, n_stocks, n_features))

@pytest.fixture
def sample_prediction_result():
    """Sample prediction result"""
    return {
        'prediction': 1,
        'probability': 0.67,
        'direction': 'UP',
        'confidence': 'Medium',
        'confidence_score': 0.34,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'test_v1.0'
    }
