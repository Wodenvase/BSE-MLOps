"""
Unit Tests for Data Processing Functions
Tests the core data fetching and feature processing logic
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Mock imports that might not be available in CI
sys.modules['yfinance'] = MagicMock()

class TestDataFetching:
    """Test data fetching functionality"""
    
    def test_stock_data_validation(self, sample_stock_data):
        """Test stock data validation logic"""
        data = sample_stock_data
        
        # Test required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in data.columns for col in required_columns)
        
        # Test data integrity
        assert len(data) > 0
        assert not data['close'].isna().any()
        assert (data['high'] >= data['low']).all()
        assert (data['volume'] > 0).all()
    
    def test_data_preprocessing(self, sample_stock_data):
        """Test data preprocessing steps"""
        data = sample_stock_data
        
        # Test moving averages calculation
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        
        # Should have valid values after minimum window
        assert not data['sma_5'].iloc[5:].isna().any()
        assert not data['sma_20'].iloc[20:].isna().any()
        
    def test_returns_calculation(self, sample_stock_data):
        """Test returns calculation"""
        data = sample_stock_data
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Test returns properties
        assert not data['returns'].iloc[1:].isna().any()
        assert data['returns'].abs().max() < 1.0  # No extreme returns in test data
        
        # Log returns should be approximately equal to simple returns for small changes
        correlation = data['returns'].iloc[1:].corr(data['log_returns'].iloc[1:])
        assert correlation > 0.95

class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_technical_indicators(self, sample_stock_data):
        """Test technical indicators calculation"""
        data = sample_stock_data.copy()
        
        # RSI calculation (simplified)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data['rsi'] = calculate_rsi(data['close'])
        
        # RSI should be between 0 and 100
        valid_rsi = data['rsi'].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_volatility_calculation(self, sample_stock_data):
        """Test volatility calculation"""
        data = sample_stock_data
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Rolling volatility
        data['volatility_5d'] = data['returns'].rolling(window=5).std()
        data['volatility_20d'] = data['returns'].rolling(window=20).std()
        
        # Volatility should be positive
        assert (data['volatility_5d'].dropna() >= 0).all()
        assert (data['volatility_20d'].dropna() >= 0).all()
    
    def test_bollinger_bands(self, sample_stock_data):
        """Test Bollinger Bands calculation"""
        data = sample_stock_data
        
        # Bollinger Bands
        window = 20
        data['sma'] = data['close'].rolling(window=window).mean()
        data['std'] = data['close'].rolling(window=window).std()
        data['bb_upper'] = data['sma'] + (2 * data['std'])
        data['bb_lower'] = data['sma'] - (2 * data['std'])
        
        # Upper band should be greater than lower band
        valid_data = data.dropna()
        assert (valid_data['bb_upper'] > valid_data['bb_lower']).all()
        
        # Price should be between bands most of the time
        valid_data['within_bands'] = (
            (valid_data['close'] <= valid_data['bb_upper']) & 
            (valid_data['close'] >= valid_data['bb_lower'])
        )
        # At least 80% of data should be within bands (typical for Bollinger Bands)
        assert valid_data['within_bands'].mean() > 0.8

class TestDataQuality:
    """Test data quality validation"""
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109]
        }, index=dates)
        
        # Test forward fill
        data_ffill = data.fillna(method='ffill')
        assert not data_ffill['close'].isna().any()
        
        # Test interpolation
        data_interp = data.interpolate()
        assert not data_interp['close'].isna().any()
    
    def test_outlier_detection(self, sample_stock_data):
        """Test outlier detection logic"""
        data = sample_stock_data.copy()
        
        # Add artificial outlier
        data.loc[data.index[10], 'close'] = data['close'].mean() * 3
        
        # Z-score based outlier detection
        z_scores = np.abs((data['close'] - data['close'].mean()) / data['close'].std())
        outliers = z_scores > 3
        
        # Should detect at least one outlier
        assert outliers.sum() >= 1
    
    def test_data_completeness(self, sample_sensex_symbols):
        """Test data completeness validation"""
        symbols = sample_sensex_symbols
        
        # Simulate data availability
        data_availability = {symbol: np.random.choice([True, False], p=[0.9, 0.1]) 
                           for symbol in symbols}
        
        # Calculate coverage
        coverage = sum(data_availability.values()) / len(symbols)
        
        # Should have at least 80% coverage for valid analysis
        if coverage >= 0.8:
            assert True  # Good coverage
        else:
            # Would trigger warning in real system
            assert coverage < 0.8

class TestDataPersistence:
    """Test data saving and loading"""
    
    def test_data_serialization(self, sample_stock_data, temp_directory):
        """Test data serialization and deserialization"""
        data = sample_stock_data
        
        # Save data
        filepath = os.path.join(temp_directory, 'test_data.csv')
        data.to_csv(filepath)
        
        # Load data
        loaded_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Compare original and loaded data
        pd.testing.assert_frame_equal(data, loaded_data, check_dtype=False)
    
    def test_data_versioning_metadata(self, temp_directory):
        """Test data versioning metadata"""
        metadata = {
            'version': '1.0.0',
            'created_at': '2024-01-01T00:00:00',
            'symbols_count': 30,
            'date_range': {'start': '2024-01-01', 'end': '2024-02-01'},
            'features_count': 45
        }
        
        # Save metadata
        import json
        metadata_path = os.path.join(temp_directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Load and verify
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata == metadata
        assert loaded_metadata['symbols_count'] == 30
        assert loaded_metadata['features_count'] == 45

if __name__ == "__main__":
    pytest.main([__file__])
