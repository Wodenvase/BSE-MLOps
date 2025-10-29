"""
Model Serving Infrastructure
Handles real-time prediction serving and data preprocessing for production
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('../')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelServer:
    """
    Production model serving infrastructure for SENSEX ConvLSTM predictions
    """
    
    def __init__(self, model_registry=None):
        """
        Initialize Model Server
        
        Args:
            model_registry: ModelRegistry instance for loading production models
        """
        self.model = None
        self.scaler = None
        self.model_info = None
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        if model_registry:
            self.registry = model_registry
        else:
            # Import here to avoid circular imports
            try:
                from model_registry import ModelRegistry
                self.registry = ModelRegistry()
            except ImportError:
                logger.warning("ModelRegistry not available, using local model loading")
                self.registry = None
        
        logger.info("Model Server initialized")
    
    def load_production_model(self) -> bool:
        """
        Load the production model and associated artifacts
        
        Returns:
            bool: True if successful
        """
        try:
            if self.registry:
                # Load from MLflow Model Registry
                self.model = self.registry.load_production_model()
                self.model_info = self.registry.get_production_model()
                
                if self.model is None:
                    logger.error("Failed to load production model from registry")
                    return False
                    
                logger.info(f"Loaded production model version {self.model_info['version']}")
            else:
                # Fallback to local model loading
                model_path = '../models/best_model.h5'
                if os.path.exists(model_path):
                    self.model = keras.models.load_model(model_path)
                    logger.info("Loaded local model")
                else:
                    logger.error("No model found locally")
                    return False
            
            # Load scaler if available
            self._load_scaler()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            return False
    
    def _load_scaler(self):
        """Load the data scaler used during training"""
        try:
            scaler_path = '../models/feature_scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            else:
                logger.warning("Feature scaler not found, using model without scaling")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
    
    def preprocess_features(self, raw_data: Dict[str, pd.DataFrame], 
                          sequence_length: int = 30) -> Optional[np.ndarray]:
        """
        Preprocess raw market data into model-ready features
        
        Args:
            raw_data: Dictionary of stock data {symbol: DataFrame}
            sequence_length: Length of input sequence
            
        Returns:
            np.ndarray: Preprocessed feature array or None
        """
        try:
            # Import feature processing utilities
            sys.path.append('../data')
            from process_features import SensexFeatureProcessor
            
            processor = SensexFeatureProcessor()
            
            # Process features for each stock
            all_features = []
            stock_symbols = sorted(raw_data.keys())
            
            for symbol in stock_symbols:
                if symbol == '^BSESN':  # Skip SENSEX index for stock features
                    continue
                    
                df = raw_data[symbol]
                if df.empty:
                    logger.error(f"Empty data for {symbol}")
                    return None
                
                # Calculate features
                features_df = processor.calculate_technical_indicators(df)
                
                if features_df.empty:
                    logger.error(f"Failed to calculate features for {symbol}")
                    return None
                
                # Get the latest sequence_length rows
                if len(features_df) < sequence_length:
                    logger.error(f"Insufficient data for {symbol}: {len(features_df)} < {sequence_length}")
                    return None
                
                latest_features = features_df.tail(sequence_length)
                all_features.append(latest_features.values)
            
            if not all_features:
                logger.error("No valid features calculated")
                return None
            
            # Stack features: (sequence_length, n_stocks, n_features)
            feature_array = np.stack(all_features, axis=1)
            
            # Add batch dimension: (1, sequence_length, n_stocks, n_features)
            feature_array = np.expand_dims(feature_array, axis=0)
            
            # Apply scaling if available
            if self.scaler is not None:
                original_shape = feature_array.shape
                # Reshape for scaling: (batch * seq * stocks, features)
                reshaped = feature_array.reshape(-1, feature_array.shape[-1])
                scaled = self.scaler.transform(reshaped)
                feature_array = scaled.reshape(original_shape)
                logger.info("Applied feature scaling")
            
            logger.info(f"Preprocessed features shape: {feature_array.shape}")
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            return None
    
    def predict(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Make prediction using the loaded model
        
        Args:
            features: Preprocessed feature array
            
        Returns:
            Dict: Prediction results or None
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return None
            
            # Make prediction
            prediction_probs = self.model.predict(features, verbose=0)
            
            # Extract probability and binary prediction
            if len(prediction_probs.shape) > 1:
                probability = float(prediction_probs[0][0])
            else:
                probability = float(prediction_probs[0])
            
            binary_prediction = 1 if probability > 0.5 else 0
            direction = "UP" if binary_prediction == 1 else "DOWN"
            
            # Calculate confidence level
            confidence_score = abs(probability - 0.5) * 2  # Scale to 0-1
            
            if confidence_score > 0.8:
                confidence = "Very High"
            elif confidence_score > 0.6:
                confidence = "High"
            elif confidence_score > 0.4:
                confidence = "Medium"
            elif confidence_score > 0.2:
                confidence = "Low"
            else:
                confidence = "Very Low"
            
            prediction_result = {
                'prediction': binary_prediction,
                'probability': probability,
                'direction': direction,
                'confidence': confidence,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_info['version'] if self.model_info else 'local'
            }
            
            logger.info(f"Prediction: {direction} ({probability:.3f}) - {confidence}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def predict_from_raw_data(self, raw_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        End-to-end prediction from raw market data
        
        Args:
            raw_data: Dictionary of stock data {symbol: DataFrame}
            
        Returns:
            Dict: Prediction results or None
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(raw_data)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if self._is_cache_valid(cached_result['timestamp']):
                    logger.info("Returning cached prediction")
                    return cached_result['prediction']
            
            # Preprocess features
            features = self.preprocess_features(raw_data)
            if features is None:
                return None
            
            # Make prediction
            prediction = self.predict(features)
            if prediction is None:
                return None
            
            # Cache result
            self.prediction_cache[cache_key] = {
                'prediction': prediction,
                'timestamp': datetime.now()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in end-to-end prediction: {str(e)}")
            return None
    
    def _generate_cache_key(self, raw_data: Dict[str, pd.DataFrame]) -> str:
        """Generate cache key from raw data"""
        try:
            # Use latest timestamp and data hash
            latest_timestamp = None
            data_hash_parts = []
            
            for symbol, df in raw_data.items():
                if not df.empty:
                    ts = df.index[-1] if hasattr(df.index[-1], 'timestamp') else str(df.index[-1])
                    if latest_timestamp is None or ts > latest_timestamp:
                        latest_timestamp = ts
                    
                    # Simple hash of latest values
                    latest_values = df.iloc[-1].values
                    data_hash_parts.append(str(hash(tuple(latest_values))))
            
            return f"{latest_timestamp}_{hash(''.join(data_hash_parts))}"
            
        except Exception:
            return str(datetime.now().timestamp())
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached result is still valid"""
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model
        
        Returns:
            Dict: Model information
        """
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'cache_size': len(self.prediction_cache)
        }
        
        if self.model_info:
            info.update({
                'version': self.model_info['version'],
                'stage': self.model_info['stage'],
                'metrics': self.model_info['metrics'],
                'creation_timestamp': self.model_info['creation_timestamp']
            })
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model server
        
        Returns:
            Dict: Health status
        """
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check model availability
        status['checks']['model_loaded'] = self.model is not None
        
        # Check if we can make a dummy prediction
        try:
            if self.model is not None:
                # Create dummy input with correct shape
                dummy_input = np.random.random((1, 30, 30, 50))  # Adjust based on actual model input shape
                _ = self.model.predict(dummy_input, verbose=0)
                status['checks']['model_inference'] = True
            else:
                status['checks']['model_inference'] = False
        except Exception as e:
            status['checks']['model_inference'] = False
            status['checks']['model_error'] = str(e)
        
        # Check cache health
        status['checks']['cache_size'] = len(self.prediction_cache)
        status['checks']['cache_healthy'] = len(self.prediction_cache) < 1000  # Prevent memory issues
        
        # Overall status
        if not all([status['checks']['model_loaded'], status['checks']['model_inference']]):
            status['status'] = 'unhealthy'
        
        return status
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def save_prediction_log(self, prediction: Dict[str, Any], 
                          request_data: Dict = None):
        """
        Save prediction to log file for monitoring
        
        Args:
            prediction: Prediction result
            request_data: Original request data (optional)
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'model_version': self.model_info['version'] if self.model_info else 'local'
            }
            
            if request_data:
                log_entry['request_data'] = request_data
            
            # Save to log file
            log_file = 'prediction_log.json'
            logs = []
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving prediction log: {str(e)}")

def main():
    """
    Test the Model Server
    """
    print("🚀 Testing Model Server...")
    
    server = ModelServer()
    
    # Load model
    print("📥 Loading production model...")
    if server.load_production_model():
        print("✅ Model loaded successfully")
        
        # Get model info
        info = server.get_model_info()
        print(f"📊 Model Info: {info}")
        
        # Health check
        health = server.health_check()
        print(f"🏥 Health Check: {health['status']}")
        
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    main()
