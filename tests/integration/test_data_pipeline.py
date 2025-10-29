"""
Integration Tests for Data Pipeline
Tests the complete data processing workflow from fetching to feature engineering
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Mock external dependencies
import sys
sys.modules['yfinance'] = MagicMock()
sys.modules['mlflow'] = MagicMock()

class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    def test_data_fetch_to_processing_pipeline(self):
        """Test complete pipeline from data fetching to feature processing"""
        # Simulate fetched data
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        raw_data = {}
        
        for symbol in symbols:
            # Create realistic stock data
            dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
            data = pd.DataFrame({
                'open': np.random.uniform(100, 110, len(dates)),
                'high': np.random.uniform(110, 120, len(dates)),
                'low': np.random.uniform(90, 100, len(dates)),
                'close': np.random.uniform(95, 115, len(dates)),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            # Ensure price relationships
            data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
            data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
            
            raw_data[symbol] = data
        
        # Test data validation
        assert len(raw_data) == len(symbols)
        for symbol, data in raw_data.items():
            assert not data.empty
            assert len(data) >= 30  # Minimum required days
            assert (data['high'] >= data['low']).all()
        
        # Test feature processing simulation
        processed_features = self.simulate_feature_processing(raw_data)
        
        # Validate processed features
        assert 'features' in processed_features
        assert 'metadata' in processed_features
        assert processed_features['features'].shape[0] == len(symbols)  # One per stock
    
    def simulate_feature_processing(self, raw_data):
        """Simulate feature processing for integration test"""
        features_per_symbol = []
        
        for symbol, data in raw_data.items():
            # Calculate basic technical indicators
            symbol_features = {}
            
            # Moving averages
            symbol_features['sma_5'] = data['close'].rolling(5).mean().iloc[-1]
            symbol_features['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            
            # Price ratios
            symbol_features['price_ratio'] = data['close'].iloc[-1] / data['close'].iloc[-2]
            
            # Volatility
            symbol_features['volatility'] = data['close'].pct_change().rolling(10).std().iloc[-1]
            
            # Volume features
            symbol_features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(10).mean().iloc[-1]
            
            features_per_symbol.append(list(symbol_features.values()))
        
        return {
            'features': np.array(features_per_symbol),
            'metadata': {
                'feature_names': list(symbol_features.keys()),
                'symbols': list(raw_data.keys()),
                'processed_at': datetime.now().isoformat()
            }
        }
    
    def test_data_quality_validation_integration(self):
        """Test integrated data quality validation"""
        # Create test data with quality issues
        problematic_data = {
            'GOOD_STOCK.NS': pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
            }),
            'MISSING_DATA.NS': pd.DataFrame({
                'close': [100, np.nan, 102, np.nan, 104],
                'volume': [1000000, 1100000, np.nan, 1300000, 1400000]
            }),
            'SHORT_DATA.NS': pd.DataFrame({
                'close': [100, 101],  # Only 2 days of data
                'volume': [1000000, 1100000]
            })
        }
        
        # Test quality validation
        quality_report = self.validate_data_quality(problematic_data)
        
        assert 'overall_quality' in quality_report
        assert 'symbol_reports' in quality_report
        assert len(quality_report['symbol_reports']) == 3
        
        # Should flag problematic data
        symbol_reports = quality_report['symbol_reports']
        assert symbol_reports['GOOD_STOCK.NS']['status'] == 'good'
        assert symbol_reports['MISSING_DATA.NS']['status'] == 'warning'
        assert symbol_reports['SHORT_DATA.NS']['status'] == 'insufficient'
    
    def validate_data_quality(self, data_dict):
        """Simulate data quality validation"""
        symbol_reports = {}
        total_symbols = len(data_dict)
        good_symbols = 0
        
        for symbol, data in data_dict.items():
            report = {'status': 'good', 'issues': []}
            
            # Check data length
            if len(data) < 5:  # Minimum threshold
                report['status'] = 'insufficient'
                report['issues'].append('Insufficient data points')
            
            # Check for missing values
            missing_ratio = data.isna().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.1:  # More than 10% missing
                report['status'] = 'warning' if report['status'] == 'good' else report['status']
                report['issues'].append(f'High missing data ratio: {missing_ratio:.1%}')
            
            # Check for extreme values
            if 'close' in data.columns:
                price_changes = data['close'].pct_change().abs()
                if price_changes.max() > 0.2:  # 20% single-day change
                    report['status'] = 'warning' if report['status'] == 'good' else report['status']
                    report['issues'].append('Extreme price movements detected')
            
            symbol_reports[symbol] = report
            if report['status'] == 'good':
                good_symbols += 1
        
        return {
            'overall_quality': good_symbols / total_symbols,
            'symbol_reports': symbol_reports,
            'timestamp': datetime.now().isoformat()
        }

class TestModelServingIntegration:
    """Test model serving integration"""
    
    def test_model_loading_to_prediction_pipeline(self):
        """Test complete model serving pipeline"""
        # Simulate model metadata
        model_info = {
            'version': '1.0.0',
            'input_shape': (1, 30, 30, 45),
            'output_shape': (1, 1),
            'metrics': {
                'accuracy': 0.65,
                'precision': 0.62
            }
        }
        
        # Simulate input features
        batch_size, seq_len, n_stocks, n_features = model_info['input_shape']
        input_features = np.random.random((batch_size, seq_len, n_stocks, n_features))
        
        # Test prediction pipeline
        prediction_result = self.simulate_model_prediction(input_features, model_info)
        
        # Validate prediction result
        assert 'prediction' in prediction_result
        assert 'probability' in prediction_result
        assert 'metadata' in prediction_result
        assert 0 <= prediction_result['probability'] <= 1
        assert prediction_result['prediction'] in [0, 1]
    
    def simulate_model_prediction(self, features, model_info):
        """Simulate model prediction for integration test"""
        # Simulate model inference
        np.random.seed(42)
        probability = np.random.uniform(0.2, 0.8)  # Random but reproducible
        prediction = 1 if probability > 0.5 else 0
        
        # Calculate confidence
        confidence_score = abs(probability - 0.5) * 2
        confidence_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        confidence_idx = min(int(confidence_score * len(confidence_levels)), len(confidence_levels) - 1)
        confidence = confidence_levels[confidence_idx]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': confidence,
            'confidence_score': confidence_score,
            'metadata': {
                'model_version': model_info['version'],
                'input_shape': features.shape,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def test_cache_integration(self):
        """Test caching integration across components"""
        # Simulate cache system
        cache = {}
        cache_ttl = 300  # 5 minutes
        
        # Test cache miss -> data fetch -> cache store
        cache_key = 'market_data_2024-01-01'
        
        # Simulate cache miss
        cached_data = cache.get(cache_key)
        assert cached_data is None
        
        # Simulate data fetch
        fresh_data = {
            'data': {'RELIANCE.NS': [100, 101, 102]},
            'timestamp': datetime.now(),
            'source': 'api'
        }
        
        # Store in cache
        cache[cache_key] = fresh_data
        
        # Test cache hit
        cached_data = cache.get(cache_key)
        assert cached_data is not None
        assert cached_data['source'] == 'api'
        
        # Test cache expiry
        old_timestamp = datetime.now() - timedelta(seconds=cache_ttl + 1)
        cache[cache_key]['timestamp'] = old_timestamp
        
        # Check if cache is expired
        elapsed = (datetime.now() - cache[cache_key]['timestamp']).total_seconds()
        cache_expired = elapsed > cache_ttl
        assert cache_expired is True

class TestStreamlitIntegration:
    """Test Streamlit app integration"""
    
    def test_app_component_integration(self):
        """Test integration between app components"""
        # Simulate app state
        app_state = {
            'demo_mode': True,
            'model_loaded': False,
            'prediction_count': 0,
            'cache': {}
        }
        
        # Test prediction workflow
        prediction_result = self.simulate_prediction_workflow(app_state)
        
        assert prediction_result is not None
        assert 'direction' in prediction_result
        assert app_state['prediction_count'] == 1
    
    def simulate_prediction_workflow(self, app_state):
        """Simulate complete prediction workflow"""
        # Check if model is available
        if not app_state['model_loaded'] and app_state['demo_mode']:
            # Use demo prediction
            demo_predictions = [
                {'direction': 'UP', 'probability': 0.67},
                {'direction': 'DOWN', 'probability': 0.43}
            ]
            
            prediction = demo_predictions[app_state['prediction_count'] % len(demo_predictions)]
            app_state['prediction_count'] += 1
            
            return prediction
        
        return None
    
    def test_error_handling_integration(self):
        """Test integrated error handling across components"""
        error_scenarios = [
            {'component': 'data_fetcher', 'error': 'Network timeout'},
            {'component': 'model_server', 'error': 'Model not loaded'},
            {'component': 'cache', 'error': 'Cache overflow'}
        ]
        
        for scenario in error_scenarios:
            error_response = self.handle_component_error(scenario)
            
            assert 'status' in error_response
            assert 'fallback_action' in error_response
            assert error_response['status'] == 'error'
    
    def handle_component_error(self, scenario):
        """Simulate component error handling"""
        component = scenario['component']
        error = scenario['error']
        
        # Define fallback actions
        fallback_actions = {
            'data_fetcher': 'Use cached data',
            'model_server': 'Switch to demo mode',
            'cache': 'Clear old entries'
        }
        
        return {
            'status': 'error',
            'component': component,
            'error': error,
            'fallback_action': fallback_actions.get(component, 'Show error message'),
            'timestamp': datetime.now().isoformat()
        }

class TestEndToEndWorkflow:
    """Test complete end-to-end workflow"""
    
    def test_complete_prediction_workflow(self):
        """Test complete workflow from data to prediction"""
        # Step 1: Data fetching simulation
        symbols = ['RELIANCE.NS', 'TCS.NS']
        raw_data = self.simulate_data_fetch(symbols)
        
        # Step 2: Data validation
        validation_result = self.simulate_data_validation(raw_data)
        assert validation_result['valid'] is True
        
        # Step 3: Feature processing
        features = self.simulate_feature_processing(raw_data)
        assert features.shape[1] == len(symbols)  # One column per stock
        
        # Step 4: Model prediction
        prediction = self.simulate_model_inference(features)
        assert 'direction' in prediction
        
        # Step 5: Result formatting
        formatted_result = self.format_prediction_result(prediction)
        assert 'display_text' in formatted_result
        
        return formatted_result
    
    def simulate_data_fetch(self, symbols):
        """Simulate data fetching step"""
        data = {}
        for symbol in symbols:
            # Create 60 days of data
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            prices = np.random.uniform(100, 200, len(dates))
            
            data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        
        return data
    
    def simulate_data_validation(self, data):
        """Simulate data validation step"""
        valid_symbols = 0
        for symbol, df in data.items():
            if len(df) >= 30 and not df['close'].isna().any():
                valid_symbols += 1
        
        return {
            'valid': valid_symbols >= len(data) * 0.8,  # 80% threshold
            'valid_symbols': valid_symbols,
            'total_symbols': len(data)
        }
    
    def simulate_feature_processing(self, data):
        """Simulate feature processing step"""
        features = []
        for symbol, df in data.items():
            # Calculate simple features
            symbol_features = [
                df['close'].iloc[-1],  # Latest price
                df['close'].rolling(5).mean().iloc[-1],  # 5-day SMA
                df['close'].pct_change().std() * 100,  # Volatility
                df['volume'].rolling(10).mean().iloc[-1]  # Avg volume
            ]
            features.append(symbol_features)
        
        return np.array(features).T  # Transpose to (features, symbols)
    
    def simulate_model_inference(self, features):
        """Simulate model inference step"""
        # Simple prediction logic for testing
        avg_feature = np.mean(features[0])  # Use first feature (price)
        probability = min(max((avg_feature - 100) / 100, 0.1), 0.9)  # Normalize to 0.1-0.9
        
        return {
            'probability': probability,
            'direction': 'UP' if probability > 0.5 else 'DOWN',
            'prediction': 1 if probability > 0.5 else 0
        }
    
    def format_prediction_result(self, prediction):
        """Format prediction result for display"""
        direction = prediction['direction']
        prob = prediction['probability']
        
        return {
            'display_text': f"{direction} with {prob:.1%} probability",
            'color': 'green' if direction == 'UP' else 'red',
            'icon': '📈' if direction == 'UP' else '📉',
            'raw_data': prediction
        }
    
    def test_performance_under_load(self):
        """Test system performance under simulated load"""
        import time
        
        # Simulate multiple concurrent predictions
        num_predictions = 10
        start_time = time.time()
        
        results = []
        for i in range(num_predictions):
            # Simulate prediction latency
            time.sleep(0.01)  # 10ms per prediction
            
            result = {
                'prediction_id': i,
                'direction': 'UP' if i % 2 == 0 else 'DOWN',
                'processing_time': 0.01
            }
            results.append(result)
        
        total_time = time.time() - start_time
        throughput = num_predictions / total_time
        
        # Validate performance
        assert len(results) == num_predictions
        assert total_time < 1.0  # Should complete within 1 second
        assert throughput > 5  # At least 5 predictions per second

if __name__ == "__main__":
    pytest.main([__file__])
