"""
Feature engineering module for creating Component Feature Maps.
Calculates technical indicators and creates 2D feature matrices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Creates Component Feature Maps from raw stock data
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize the feature engineer
        
        Args:
            symbols: List of stock symbols
        """
        self.symbols = symbols
        self.feature_names = []
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a single stock
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            df = df.copy()
            
            # Basic returns and volatility
            df['returns_1d'] = df['close'].pct_change()
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_10d'] = df['close'].pct_change(10)
            df['volatility_5d'] = df['returns_1d'].rolling(5).std()
            df['volatility_20d'] = df['returns_1d'].rolling(20).std()
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # RSI
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
            
            # MACD
            macd_ind = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()
            df['macd_histogram'] = macd_ind.macd_diff()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # Price relative to moving averages
            df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
            
            # Bollinger Bands
            bb_ind = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_ind.bollinger_hband()
            df['bb_lower'] = bb_ind.bollinger_lband()
            df['bb_middle'] = bb_ind.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic Oscillator
            stoch_ind = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch_ind.stoch()
            df['stoch_d'] = stoch_ind.stoch_signal()
            
            # ADX (Average Directional Index)
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
            
            # Rate of Change
            df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            
            # Momentum
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Average True Range
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # On-Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_sma'] = df['obv'].rolling(20).mean()
            df['obv_ratio'] = df['obv'] / df['obv_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def select_features_for_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and normalize features for the component feature map
        
        Args:
            df: DataFrame with all calculated indicators
            
        Returns:
            DataFrame with selected and normalized features
        """
        # Define the features to include in the component map
        feature_columns = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_20d',
            'volume_ratio',
            'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'price_vs_sma20', 'price_vs_sma50',
            'bb_width', 'bb_position',
            'stoch_k', 'stoch_d',
            'adx', 'cci', 'williams_r',
            'roc_10', 'momentum_10',
            'atr', 'obv_ratio'
        ]
        
        # Store feature names for later reference
        self.feature_names = feature_columns
        
        # Select features
        features_df = df[feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='bfill').fillna(method='ffill')
        
        # Normalize features to [-1, 1] range using tanh scaling
        normalized_features = features_df.copy()
        
        for col in feature_columns:
            if col in features_df.columns:
                # Calculate rolling statistics for normalization
                mean = features_df[col].rolling(252, min_periods=50).mean()  # 1 year rolling mean
                std = features_df[col].rolling(252, min_periods=50).std()    # 1 year rolling std
                
                # Z-score normalization followed by tanh scaling
                z_score = (features_df[col] - mean) / (std + 1e-8)
                normalized_features[col] = np.tanh(z_score)
        
        return normalized_features
    
    def create_component_feature_maps(self, 
                                    stock_data: Dict[str, pd.DataFrame],
                                    index_symbol: str) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Create Component Feature Maps (30 x k matrices) for each day
        
        Args:
            stock_data: Dictionary mapping symbol to DataFrame
            index_symbol: SENSEX index symbol for target creation
            
        Returns:
            Tuple of (feature_maps, targets, dates)
        """
        logger.info("Creating Component Feature Maps...")
        
        # Process each stock to get features
        processed_stocks = {}
        
        for symbol in self.symbols:
            if symbol in stock_data:
                logger.info(f"Processing features for {symbol}")
                
                # Calculate technical indicators
                df_with_indicators = self.calculate_technical_indicators(stock_data[symbol])
                
                # Select and normalize features
                features = self.select_features_for_map(df_with_indicators)
                
                processed_stocks[symbol] = features
            else:
                logger.warning(f"No data found for {symbol}")
        
        # Get common date range
        common_dates = None
        for df in processed_stocks.values():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        logger.info(f"Common date range: {len(common_dates)} days")
        
        # Create 3D array: (n_days, n_stocks, n_features)
        n_days = len(common_dates)
        n_stocks = len(processed_stocks)
        n_features = len(self.feature_names)
        
        feature_maps = np.zeros((n_days, n_stocks, n_features))
        
        # Fill the feature maps
        for stock_idx, (symbol, features_df) in enumerate(processed_stocks.items()):
            for date_idx, date in enumerate(common_dates):
                if date in features_df.index:
                    feature_maps[date_idx, stock_idx, :] = features_df.loc[date].values
                else:
                    # Use previous day's values if current day is missing
                    if date_idx > 0:
                        feature_maps[date_idx, stock_idx, :] = feature_maps[date_idx-1, stock_idx, :]
        
        # Create targets from SENSEX index
        targets = self.create_targets(stock_data[index_symbol], common_dates)
        
        logger.info(f"Created feature maps shape: {feature_maps.shape}")
        logger.info(f"Created targets shape: {targets.shape}")
        
        return feature_maps, targets, common_dates
    
    def create_targets(self, index_data: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Create binary targets (1 for up, 0 for down) from SENSEX index data
        
        Args:
            index_data: SENSEX index DataFrame
            dates: Common dates for alignment
            
        Returns:
            Binary target array
        """
        # Align index data with common dates
        aligned_index = index_data.reindex(dates)
        
        # Calculate next-day returns
        next_day_returns = aligned_index['close'].shift(-1) / aligned_index['close'] - 1
        
        # Create binary targets (1 for positive returns, 0 for negative)
        targets = (next_day_returns > 0).astype(int)
        
        # Remove the last day since we can't predict beyond our data
        targets = targets[:-1].values
        
        return targets
    
    def save_feature_maps(self, 
                         feature_maps: np.ndarray, 
                         targets: np.ndarray, 
                         dates: pd.DatetimeIndex, 
                         output_dir: str) -> None:
        """
        Save feature maps and targets to files
        
        Args:
            feature_maps: 3D feature array
            targets: Target array
            dates: Date index
            output_dir: Output directory
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save feature maps
        np.save(os.path.join(output_dir, 'feature_maps.npy'), feature_maps)
        
        # Save targets
        np.save(os.path.join(output_dir, 'targets.npy'), targets)
        
        # Save dates
        dates_df = pd.DataFrame({'date': dates[:-1]})  # Exclude last date (no target)
        dates_df.to_csv(os.path.join(output_dir, 'dates.csv'), index=False)
        
        # Save feature names
        feature_names_df = pd.DataFrame({
            'feature_index': range(len(self.feature_names)),
            'feature_name': self.feature_names
        })
        feature_names_df.to_csv(os.path.join(output_dir, 'feature_names.csv'), index=False)
        
        # Save stock names
        stock_names_df = pd.DataFrame({
            'stock_index': range(len(self.symbols)),
            'stock_symbol': self.symbols
        })
        stock_names_df.to_csv(os.path.join(output_dir, 'stock_names.csv'), index=False)
        
        logger.info(f"Saved feature maps to {output_dir}")
        logger.info(f"Feature maps shape: {feature_maps.shape}")
        logger.info(f"Targets shape: {targets.shape}")


def main():
    """
    Main function to create feature maps
    """
    # Import configuration
    from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX, PATHS
    from src.data.fetch_data import SensexDataFetcher
    
    # Initialize components
    fetcher = SensexDataFetcher(SENSEX_30_SYMBOLS, SENSEX_INDEX)
    engineer = FeatureEngineer(SENSEX_30_SYMBOLS)
    
    # Load raw data
    logger.info("Loading raw data...")
    stock_data = fetcher.load_raw_data(PATHS['raw_data'])
    
    if not stock_data:
        logger.error("No stock data found. Please run fetch_data.py first.")
        return
    
    # Create feature maps
    feature_maps, targets, dates = engineer.create_component_feature_maps(
        stock_data, SENSEX_INDEX
    )
    
    # Save processed data
    engineer.save_feature_maps(feature_maps, targets, dates, PATHS['processed_data'])
    
    logger.info("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
