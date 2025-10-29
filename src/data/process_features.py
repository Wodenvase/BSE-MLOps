"""
Advanced Feature Map Processor for SENSEX Components
Transforms raw price data into (num_days, 30, k_features) NumPy arrays with comprehensive technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import ta
import logging
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureProcessingError(Exception):
    """Custom exception for feature processing issues"""
    pass

class AdvancedFeatureProcessor:
    """
    Processes raw stock data into comprehensive feature maps for ConvLSTM
    """
    
    def __init__(self, symbols: List[str], target_features: int = 40):
        """
        Initialize the feature processor
        
        Args:
            symbols: List of stock symbols (should be 30 for SENSEX)
            target_features: Target number of features per stock
        """
        self.symbols = symbols
        self.target_features = target_features
        self.feature_names = []
        self.feature_configs = self._get_feature_configs()
        self.processed_data = {}
        
        logger.info(f"Initialized AdvancedFeatureProcessor for {len(symbols)} symbols")
        logger.info(f"Target features per stock: {target_features}")
    
    def _get_feature_configs(self) -> Dict:
        """
        Define comprehensive technical indicator configurations
        """
        return {
            # Price-based features
            'price_features': {
                'returns_1d': {'type': 'returns', 'periods': 1},
                'returns_3d': {'type': 'returns', 'periods': 3},
                'returns_5d': {'type': 'returns', 'periods': 5},
                'returns_10d': {'type': 'returns', 'periods': 10},
                'log_returns_1d': {'type': 'log_returns', 'periods': 1},
                'price_change_pct': {'type': 'price_change_pct'},
                'gap_up_down': {'type': 'gap'},
                'typical_price': {'type': 'typical_price'},
            },
            
            # Volatility features
            'volatility_features': {
                'volatility_5d': {'type': 'volatility', 'window': 5},
                'volatility_10d': {'type': 'volatility', 'window': 10},
                'volatility_20d': {'type': 'volatility', 'window': 20},
                'parkinson_volatility': {'type': 'parkinson_vol', 'window': 20},
                'atr_14': {'type': 'atr', 'window': 14},
                'atr_ratio': {'type': 'atr_ratio', 'window': 14},
            },
            
            # Moving averages and trends
            'trend_features': {
                'sma_5': {'type': 'sma', 'window': 5},
                'sma_10': {'type': 'sma', 'window': 10},
                'sma_20': {'type': 'sma', 'window': 20},
                'sma_50': {'type': 'sma', 'window': 50},
                'ema_12': {'type': 'ema', 'window': 12},
                'ema_26': {'type': 'ema', 'window': 26},
                'price_vs_sma20': {'type': 'price_vs_ma', 'window': 20},
                'price_vs_sma50': {'type': 'price_vs_ma', 'window': 50},
                'ma_convergence': {'type': 'ma_convergence'},
            },
            
            # Momentum indicators
            'momentum_features': {
                'rsi_14': {'type': 'rsi', 'window': 14},
                'rsi_21': {'type': 'rsi', 'window': 21},
                'stoch_k': {'type': 'stoch_k', 'k_window': 14, 'd_window': 3},
                'stoch_d': {'type': 'stoch_d', 'k_window': 14, 'd_window': 3},
                'williams_r': {'type': 'williams_r', 'window': 14},
                'roc_10': {'type': 'roc', 'window': 10},
                'momentum_10': {'type': 'momentum', 'window': 10},
                'cci_20': {'type': 'cci', 'window': 20},
            },
            
            # MACD family
            'macd_features': {
                'macd': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
                'macd_signal': {'type': 'macd_signal', 'fast': 12, 'slow': 26, 'signal': 9},
                'macd_histogram': {'type': 'macd_histogram', 'fast': 12, 'slow': 26, 'signal': 9},
                'macd_crossover': {'type': 'macd_crossover'},
            },
            
            # Bollinger Bands
            'bollinger_features': {
                'bb_upper': {'type': 'bb_upper', 'window': 20, 'std': 2},
                'bb_lower': {'type': 'bb_lower', 'window': 20, 'std': 2},
                'bb_middle': {'type': 'bb_middle', 'window': 20},
                'bb_width': {'type': 'bb_width', 'window': 20, 'std': 2},
                'bb_position': {'type': 'bb_position', 'window': 20, 'std': 2},
                'bb_squeeze': {'type': 'bb_squeeze', 'window': 20},
            },
            
            # Volume features
            'volume_features': {
                'volume_sma_20': {'type': 'volume_sma', 'window': 20},
                'volume_ratio': {'type': 'volume_ratio', 'window': 20},
                'volume_rate_change': {'type': 'volume_roc', 'window': 5},
                'obv': {'type': 'obv'},
                'obv_sma': {'type': 'obv_sma', 'window': 20},
                'volume_price_trend': {'type': 'vpt'},
            },
            
            # Advanced indicators
            'advanced_features': {
                'adx_14': {'type': 'adx', 'window': 14},
                'di_plus': {'type': 'di_plus', 'window': 14},
                'di_minus': {'type': 'di_minus', 'window': 14},
                'ichimoku_a': {'type': 'ichimoku_a'},
                'ichimoku_b': {'type': 'ichimoku_b'},
            }
        }
    
    def calculate_comprehensive_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for a single stock
        
        Args:
            df: Stock OHLCV DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            DataFrame with calculated features
        """
        logger.debug(f"Calculating features for {symbol}")
        
        try:
            # Make a copy to avoid modifying original data
            features_df = df.copy()
            
            # Calculate all feature categories
            features_df = self._calculate_price_features(features_df)
            features_df = self._calculate_volatility_features(features_df)
            features_df = self._calculate_trend_features(features_df)
            features_df = self._calculate_momentum_features(features_df)
            features_df = self._calculate_macd_features(features_df)
            features_df = self._calculate_bollinger_features(features_df)
            features_df = self._calculate_volume_features(features_df)
            features_df = self._calculate_advanced_features(features_df)
            
            logger.debug(f"✓ Calculated {len(features_df.columns)} total features for {symbol}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {str(e)}")
            raise FeatureProcessingError(f"Feature calculation failed for {symbol}: {str(e)}")
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        # Returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_3d'] = df['close'].pct_change(3)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['log_returns_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        df['gap_up_down'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        # Rolling volatility
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        
        # Parkinson volatility (using high-low)
        df['parkinson_volatility'] = np.sqrt(
            (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        )
        
        # Average True Range
        df['atr_14'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        return df
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based features"""
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = ta.trend.SMAIndicator(df['close'], window=window).sma_indicator()
        
        # Exponential Moving Averages
        for window in [12, 26]:
            df[f'ema_{window}'] = ta.trend.EMAIndicator(df['close'], window=window).ema_indicator()
        
        # Price vs Moving Averages
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Moving Average Convergence
        df['ma_convergence'] = (df['sma_10'] - df['sma_20']) / df['sma_20']
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
        
        # Rate of Change
        df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
        
        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Commodity Channel Index
        df['cci_20'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        return df
    
    def _calculate_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD family indicators"""
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # MACD crossover signal
        df['macd_crossover'] = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
            np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0)
        )
        
        return df
    
    def _calculate_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands features"""
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        # Bollinger Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band squeeze
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean()
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        # Volume moving average and ratio
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_rate_change'] = df['volume'].pct_change(5)
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # Volume Price Trend
        df['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        return df
    
    def _calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        # ADX (Average Directional Index)
        df['adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['di_plus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx_pos()
        df['di_minus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx_neg()
        
        # Ichimoku Cloud components
        try:
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
        except:
            df['ichimoku_a'] = np.nan
            df['ichimoku_b'] = np.nan
        
        return df
    
    def select_top_features(self, features_df: pd.DataFrame, method: str = 'variance') -> List[str]:
        """
        Select top features based on various criteria
        
        Args:
            features_df: DataFrame with all calculated features
            method: Selection method ('variance', 'correlation', 'importance')
            
        Returns:
            List of selected feature names
        """
        # Get numeric columns only (excluding OHLCV and basic info)
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'adj close']
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col.lower() not in exclude_columns]
        
        if method == 'variance':
            # Select features with highest variance (after normalization)
            feature_data = features_df[feature_columns].fillna(method='bfill').fillna(method='ffill')
            
            # Normalize features
            normalized_data = (feature_data - feature_data.mean()) / feature_data.std()
            
            # Calculate variance
            variances = normalized_data.var().sort_values(ascending=False)
            
            # Select top features
            selected_features = variances.head(self.target_features).index.tolist()
            
        elif method == 'correlation':
            # Select features with diverse correlations
            feature_data = features_df[feature_columns].fillna(method='bfill').fillna(method='ffill')
            
            corr_matrix = feature_data.corr().abs()
            
            # Use a simple greedy selection to minimize correlation
            selected_features = []
            remaining_features = feature_columns.copy()
            
            # Start with feature having highest variance
            variances = feature_data.var().sort_values(ascending=False)
            selected_features.append(variances.index[0])
            remaining_features.remove(variances.index[0])
            
            while len(selected_features) < self.target_features and remaining_features:
                # Find feature with lowest average correlation to selected features
                avg_correlations = {}
                for feature in remaining_features:
                    avg_corr = corr_matrix.loc[feature, selected_features].mean()
                    avg_correlations[feature] = avg_corr
                
                # Select feature with lowest average correlation
                next_feature = min(avg_correlations, key=avg_correlations.get)
                selected_features.append(next_feature)
                remaining_features.remove(next_feature)
        
        else:  # Default to all available features
            selected_features = feature_columns[:self.target_features]
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features
    
    def create_feature_maps(self, 
                           stock_data: Dict[str, pd.DataFrame],
                           index_symbol: str,
                           feature_selection_method: str = 'variance') -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
        """
        Create comprehensive feature maps from stock data
        
        Args:
            stock_data: Dictionary mapping symbol to DataFrame
            index_symbol: SENSEX index symbol for target creation
            feature_selection_method: Method for feature selection
            
        Returns:
            Tuple of (feature_maps, targets, dates, feature_names)
        """
        logger.info("Creating comprehensive feature maps...")
        logger.info(f"Processing {len(stock_data)} symbols")
        
        # Process each stock to calculate features
        processed_stocks = {}
        all_feature_names = set()
        
        for symbol in self.symbols:
            if symbol in stock_data:
                logger.info(f"Processing features for {symbol}")
                
                # Calculate comprehensive features
                features_df = self.calculate_comprehensive_features(stock_data[symbol], symbol)
                
                # Collect all feature names
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close']
                feature_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]
                all_feature_names.update(feature_cols)
                
                processed_stocks[symbol] = features_df
                
            else:
                logger.warning(f"No data found for {symbol}")
        
        # Select optimal features
        if processed_stocks:
            # Use first stock to determine feature selection
            first_stock_data = next(iter(processed_stocks.values()))
            selected_features = self.select_top_features(first_stock_data, feature_selection_method)
            self.feature_names = selected_features
        else:
            raise FeatureProcessingError("No stocks processed successfully")
        
        # Get common date range
        common_dates = None
        for df in processed_stocks.values():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        logger.info(f"Common date range: {len(common_dates)} days")
        logger.info(f"Selected features: {len(self.feature_names)}")
        
        # Create 3D feature array: (n_days, n_stocks, n_features)
        n_days = len(common_dates)
        n_stocks = len(processed_stocks)
        n_features = len(self.feature_names)
        
        feature_maps = np.zeros((n_days, n_stocks, n_features))
        
        # Fill the feature maps
        for stock_idx, (symbol, features_df) in enumerate(processed_stocks.items()):
            logger.debug(f"Filling feature map for {symbol} (index {stock_idx})")
            
            for date_idx, date in enumerate(common_dates):
                if date in features_df.index:
                    # Extract selected features for this date
                    feature_values = []
                    for feature_name in self.feature_names:
                        if feature_name in features_df.columns:
                            value = features_df.loc[date, feature_name]
                            # Handle NaN values
                            if pd.isna(value):
                                # Use previous value or 0
                                if date_idx > 0:
                                    value = feature_maps[date_idx-1, stock_idx, len(feature_values)]
                                else:
                                    value = 0.0
                            feature_values.append(value)
                        else:
                            feature_values.append(0.0)
                    
                    feature_maps[date_idx, stock_idx, :] = feature_values
                else:
                    # Use previous day's values if current day is missing
                    if date_idx > 0:
                        feature_maps[date_idx, stock_idx, :] = feature_maps[date_idx-1, stock_idx, :]
        
        # Create targets from SENSEX index
        targets = self._create_targets(stock_data[index_symbol], common_dates)
        
        # Apply normalization to feature maps
        feature_maps = self._normalize_features(feature_maps)
        
        logger.info("="*60)
        logger.info("FEATURE MAP CREATION SUMMARY")
        logger.info("="*60)
        logger.info(f"✓ Feature maps shape: {feature_maps.shape}")
        logger.info(f"✓ Targets shape: {targets.shape}")
        logger.info(f"✓ Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
        logger.info(f"✓ Features per stock: {n_features}")
        logger.info(f"✓ Total feature dimensionality: {n_stocks * n_features}")
        
        return feature_maps, targets, common_dates, self.feature_names
    
    def _create_targets(self, index_data: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Create binary targets from SENSEX index data
        
        Args:
            index_data: SENSEX index DataFrame
            dates: Common dates for alignment
            
        Returns:
            Binary target array (1 for up, 0 for down)
        """
        # Align index data with common dates
        aligned_index = index_data.reindex(dates)
        
        # Calculate next-day returns
        next_day_returns = aligned_index['close'].shift(-1) / aligned_index['close'] - 1
        
        # Create binary targets (1 for positive returns, 0 for negative)
        targets = (next_day_returns > 0).astype(int)
        
        # Remove the last day since we can't predict beyond our data
        targets = targets[:-1].values
        
        logger.info(f"Target distribution - Up: {np.sum(targets)} ({np.mean(targets):.2%}), Down: {len(targets) - np.sum(targets)} ({1-np.mean(targets):.2%})")
        
        return targets
    
    def _normalize_features(self, feature_maps: np.ndarray) -> np.ndarray:
        """
        Normalize features using robust scaling
        
        Args:
            feature_maps: Raw feature maps
            
        Returns:
            Normalized feature maps
        """
        logger.info("Applying feature normalization...")
        
        normalized_maps = feature_maps.copy()
        n_days, n_stocks, n_features = feature_maps.shape
        
        # Normalize each feature across all stocks and time
        for feature_idx in range(n_features):
            # Get all values for this feature across all stocks and time
            feature_values = feature_maps[:, :, feature_idx].flatten()
            feature_values = feature_values[~np.isnan(feature_values)]  # Remove NaN values
            
            if len(feature_values) > 0:
                # Use robust scaling (median and IQR)
                median = np.median(feature_values)
                q75, q25 = np.percentile(feature_values, [75, 25])
                iqr = q75 - q25
                
                if iqr > 0:
                    # Apply robust scaling
                    normalized_maps[:, :, feature_idx] = (feature_maps[:, :, feature_idx] - median) / iqr
                    
                    # Clip extreme outliers
                    normalized_maps[:, :, feature_idx] = np.clip(
                        normalized_maps[:, :, feature_idx], -5, 5
                    )
                else:
                    # If no variation, set to zero
                    normalized_maps[:, :, feature_idx] = 0
        
        # Handle any remaining NaN values
        normalized_maps = np.nan_to_num(normalized_maps, nan=0.0, posinf=5.0, neginf=-5.0)
        
        logger.info("✓ Feature normalization completed")
        return normalized_maps
    
    def save_feature_maps(self, 
                         feature_maps: np.ndarray,
                         targets: np.ndarray,
                         dates: pd.DatetimeIndex,
                         feature_names: List[str],
                         output_dir: str = "data/processed") -> None:
        """
        Save feature maps and metadata to files
        
        Args:
            feature_maps: 3D feature array
            targets: Target array
            dates: Date index
            feature_names: List of feature names
            output_dir: Output directory
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save feature maps and targets
        np.save(os.path.join(output_dir, 'feature_maps.npy'), feature_maps)
        np.save(os.path.join(output_dir, 'targets.npy'), targets)
        
        # Save dates (excluding last date as it has no target)
        dates_df = pd.DataFrame({
            'date': dates[:-1],  # Exclude last date
            'date_str': dates[:-1].strftime('%Y-%m-%d')
        })
        dates_df.to_csv(os.path.join(output_dir, 'dates.csv'), index=False)
        
        # Save feature metadata
        feature_metadata = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'stock_symbols': self.symbols,
            'stock_count': len(self.symbols),
            'total_features': len(self.symbols) * len(feature_names),
            'feature_map_shape': feature_maps.shape,
            'targets_shape': targets.shape,
            'creation_timestamp': datetime.now().isoformat(),
            'feature_configs': self.feature_configs
        }
        
        with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)
        
        # Save individual files for compatibility
        pd.DataFrame({'feature_name': feature_names}).to_csv(
            os.path.join(output_dir, 'feature_names.csv'), index=False
        )
        
        pd.DataFrame({'stock_symbol': self.symbols}).to_csv(
            os.path.join(output_dir, 'stock_symbols.csv'), index=False
        )
        
        logger.info("="*50)
        logger.info("SAVED FILES SUMMARY")
        logger.info("="*50)
        logger.info(f"✓ feature_maps.npy: {feature_maps.shape}")
        logger.info(f"✓ targets.npy: {targets.shape}")
        logger.info(f"✓ dates.csv: {len(dates)-1} dates")
        logger.info(f"✓ feature_metadata.json: Complete metadata")
        logger.info(f"✓ Output directory: {output_dir}")


def main():
    """
    Main function to run the feature processing pipeline
    """
    # Import dependencies
    from get_sensex_tickers import SensexTickerScraper
    from fetch_data import SensexDataFetcher
    
    logger.info("="*60)
    logger.info("STARTING ADVANCED FEATURE PROCESSING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Get SENSEX tickers
    logger.info("Step 1: Getting SENSEX component tickers...")
    scraper = SensexTickerScraper()
    tickers = scraper.get_sensex_components(validate=False)  # Skip validation for speed
    
    if len(tickers) < 25:
        logger.warning(f"Only got {len(tickers)} tickers, using fallback list")
        tickers = scraper.fallback_tickers
    
    logger.info(f"✓ Using {len(tickers)} SENSEX component tickers")
    
    # Step 2: Fetch stock data
    logger.info("Step 2: Fetching stock data...")
    index_symbol = "^BSESN"
    fetcher = SensexDataFetcher(tickers, index_symbol, max_workers=8)
    
    # Fetch data for the last 2 years
    stock_data = fetcher.fetch_all_stocks_data(
        period="2y",
        interval="1d",
        parallel=True
    )
    
    if not stock_data:
        logger.error("No stock data fetched. Exiting.")
        return
    
    # Step 3: Process features
    logger.info("Step 3: Processing comprehensive features...")
    processor = AdvancedFeatureProcessor(tickers, target_features=40)
    
    feature_maps, targets, dates, feature_names = processor.create_feature_maps(
        stock_data,
        index_symbol,
        feature_selection_method='variance'
    )
    
    # Step 4: Save results
    logger.info("Step 4: Saving feature maps...")
    processor.save_feature_maps(
        feature_maps, targets, dates, feature_names
    )
    
    logger.info("="*60)
    logger.info("ADVANCED FEATURE PROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
