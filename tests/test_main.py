"""
Basic unit tests for the SENSEX MLOps project
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.fetch_data import SensexDataFetcher
from data.create_feature_maps import FeatureEngineer
from models.convlstm_model import SensexConvLSTM
from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX, MODEL_CONFIG

class TestDataFetcher(unittest.TestCase):
    """Test cases for data fetching functionality"""
    
    def setUp(self):
        self.symbols = SENSEX_30_SYMBOLS[:5]  # Use first 5 symbols for testing
        self.fetcher = SensexDataFetcher(self.symbols, SENSEX_INDEX)
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.fetcher.symbols, self.symbols)
        self.assertEqual(self.fetcher.index_symbol, SENSEX_INDEX)
        self.assertEqual(self.fetcher.data_cache, {})
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data(self, mock_ticker):
        """Test fetching stock data"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107], 
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        result = self.fetcher.fetch_stock_data('TEST.NS')
        
        self.assertFalse(result.empty)
        self.assertIn('returns', result.columns)
        self.assertIn('symbol', result.columns)
        self.assertEqual(result['symbol'].iloc[0], 'TEST.NS')
    
    def test_get_date_range_empty_data(self):
        """Test date range with empty data"""
        start, end = self.fetcher.get_date_range({})
        self.assertIsNone(start)
        self.assertIsNone(end)


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for feature engineering"""
    
    def setUp(self):
        self.symbols = SENSEX_30_SYMBOLS[:3]
        self.engineer = FeatureEngineer(self.symbols)
        
        # Create sample data
        self.sample_data = {}
        for symbol in self.symbols:
            self.sample_data[symbol] = pd.DataFrame({
                'open': np.random.randn(100) * 10 + 50000,
                'high': np.random.randn(100) * 10 + 50500,
                'low': np.random.randn(100) * 10 + 49500,
                'close': np.random.randn(100) * 10 + 50000,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=pd.date_range('2023-01-01', periods=100))
        
        # Add index data
        self.sample_data[SENSEX_INDEX] = self.sample_data[self.symbols[0]].copy()
    
    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.engineer.symbols, self.symbols)
        self.assertEqual(self.engineer.feature_names, [])
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation"""
        df = self.sample_data[self.symbols[0]].copy()
        result = self.engineer.calculate_technical_indicators(df)
        
        # Check if new indicators are added
        self.assertIn('rsi_14', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('sma_20', result.columns)
        self.assertIn('returns_1d', result.columns)
        
        # Check data types
        self.assertTrue(result['rsi_14'].dtype in [np.float32, np.float64])
    
    def test_create_component_feature_maps(self):
        """Test feature map creation"""
        feature_maps, targets, dates = self.engineer.create_component_feature_maps(
            self.sample_data, SENSEX_INDEX
        )
        
        # Check shapes
        self.assertEqual(len(feature_maps.shape), 3)
        self.assertEqual(feature_maps.shape[1], len(self.symbols))
        self.assertEqual(len(targets.shape), 1)
        self.assertEqual(feature_maps.shape[0], len(dates))
        
        # Check target values (should be 0 or 1)
        self.assertTrue(np.all((targets == 0) | (targets == 1)))


class TestConvLSTMModel(unittest.TestCase):
    """Test cases for ConvLSTM model"""
    
    def setUp(self):
        self.input_shape = (30, 30, 24)  # (seq_len, n_stocks, n_features)
        self.model_config = MODEL_CONFIG.copy()
        self.convlstm = SensexConvLSTM(self.input_shape, self.model_config)
    
    def test_init(self):
        """Test model initialization"""
        self.assertEqual(self.convlstm.input_shape, self.input_shape)
        self.assertEqual(self.convlstm.config, self.model_config)
        self.assertIsNone(self.convlstm.model)
    
    def test_prepare_data(self):
        """Test data preparation for ConvLSTM"""
        # Create sample feature maps and targets
        feature_maps = np.random.randn(100, 30, 24)
        targets = np.random.randint(0, 2, 100)
        
        X, y = self.convlstm.prepare_data(feature_maps, targets, sequence_length=30)
        
        # Check shapes
        expected_samples = 100 - 30  # 70 samples
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], 30)  # sequence length
        self.assertEqual(X.shape[2], 30)  # n_stocks
        self.assertEqual(X.shape[3], 24)  # n_features  
        self.assertEqual(X.shape[4], 1)   # channel dimension
        self.assertEqual(y.shape[0], expected_samples)
    
    def test_split_data(self):
        """Test data splitting"""
        X = np.random.randn(100, 30, 30, 24, 1)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.convlstm.split_data(
            X, y, train_ratio=0.7, val_ratio=0.2
        )
        
        # Check sizes
        self.assertEqual(X_train.shape[0], 70)
        self.assertEqual(X_val.shape[0], 20)
        self.assertEqual(X_test.shape[0], 10)
        self.assertEqual(y_train.shape[0], 70)
        self.assertEqual(y_val.shape[0], 20)
        self.assertEqual(y_test.shape[0], 10)


class TestConfigValues(unittest.TestCase):
    """Test configuration values"""
    
    def test_sensex_symbols(self):
        """Test SENSEX symbols configuration"""
        self.assertEqual(len(SENSEX_30_SYMBOLS), 30)
        self.assertTrue(all(symbol.endswith('.NS') for symbol in SENSEX_30_SYMBOLS))
        self.assertEqual(SENSEX_INDEX, '^BSESN')
    
    def test_model_config(self):
        """Test model configuration"""
        self.assertIn('sequence_length', MODEL_CONFIG)
        self.assertIn('n_features', MODEL_CONFIG)
        self.assertIn('n_stocks', MODEL_CONFIG)
        self.assertIn('batch_size', MODEL_CONFIG)
        self.assertIn('epochs', MODEL_CONFIG)
        
        # Check reasonable values
        self.assertGreater(MODEL_CONFIG['sequence_length'], 0)
        self.assertGreater(MODEL_CONFIG['n_features'], 0)
        self.assertEqual(MODEL_CONFIG['n_stocks'], 30)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_data_validation(self):
        """Test data validation logic"""
        # Test valid data
        valid_data = np.random.randn(100, 30, 24)
        self.assertFalse(np.any(np.isnan(valid_data)))
        
        # Test data with NaN
        invalid_data = valid_data.copy()
        invalid_data[0, 0, 0] = np.nan
        self.assertTrue(np.any(np.isnan(invalid_data)))
    
    def test_array_shapes(self):
        """Test array shape validations"""
        # Test 3D array
        data_3d = np.random.randn(100, 30, 24)
        self.assertEqual(len(data_3d.shape), 3)
        
        # Test 2D array
        data_2d = np.random.randn(100, 30)
        self.assertEqual(len(data_2d.shape), 2)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataFetcher,
        TestFeatureEngineer, 
        TestConvLSTMModel,
        TestConfigValues,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
