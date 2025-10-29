"""
Enhanced Data Fetching Pipeline for SENSEX 30 Component Stocks
Fetches historical stock data with robust error handling, retry logic, and validation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_fetching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataQualityError(Exception):
    """Custom exception for data quality issues"""
    pass

class DataFetchError(Exception):
    """Custom exception for data fetching issues"""
    pass

class SensexDataFetcher:
    """
    Fetches and manages historical stock data for SENSEX 30 components
    """
    
    def __init__(self, symbols: List[str], index_symbol: str, max_workers: int = 5):
        """
        Initialize the enhanced data fetcher
        
        Args:
            symbols: List of stock symbols for SENSEX 30 components
            index_symbol: SENSEX index symbol
            max_workers: Maximum number of concurrent workers for parallel fetching
        """
        self.symbols = symbols
        self.index_symbol = index_symbol
        self.max_workers = max_workers
        self.data_cache = {}
        self.failed_tickers = []
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 1,
            'retry_delay': 2
        }
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def fetch_stock_data(self, 
                        symbol: str, 
                        period: str = "5y", 
                        interval: str = "1d",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for a single stock with retry logic
        
        Args:
            symbol: Stock symbol
            period: Data period (e.g., "5y", "1y", "6mo")
            interval: Data interval (e.g., "1d", "1h")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        max_retries = self.retry_config['max_retries']
        retry_delay = self.retry_config['retry_delay']
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching {symbol} (attempt {attempt + 1}/{max_retries})")
                
                ticker = yf.Ticker(symbol, session=self.session)
                
                # Fetch data based on parameters
                if start_date and end_date:
                    data = ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    raise DataFetchError(f"No data returned for {symbol}")
                
                # Validate data quality
                self._validate_stock_data(data, symbol)
                
                # Clean and process data
                data = self._clean_stock_data(data, symbol)
                
                logger.info(f"✓ Successfully fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"✗ Failed to fetch {symbol} after {max_retries} attempts")
                    self.failed_tickers.append(symbol)
        
        return pd.DataFrame()
    
    def _validate_stock_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate the quality of fetched stock data
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol for logging
            
        Raises:
            DataQualityError: If data quality issues are found
        """
        if data.empty:
            raise DataQualityError(f"Empty data for {symbol}")
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataQualityError(f"Missing columns for {symbol}: {missing_columns}")
        
        # Check for reasonable data
        if len(data) < 10:
            raise DataQualityError(f"Insufficient data points for {symbol}: {len(data)}")
        
        # Check for invalid prices (negative or zero)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (data[col] <= 0).any():
                raise DataQualityError(f"Invalid prices found in {col} for {symbol}")
        
        # Check price logic (High >= Low, etc.)
        if (data['High'] < data['Low']).any():
            raise DataQualityError(f"Invalid OHLC logic for {symbol}: High < Low")
        
        if (data['High'] < data['Close']).any() or (data['Low'] > data['Close']).any():
            raise DataQualityError(f"Invalid OHLC logic for {symbol}: Close outside High-Low range")
        
        # Check for excessive missing values
        missing_ratio = data.isnull().sum().max() / len(data)
        if missing_ratio > 0.05:  # More than 5% missing
            raise DataQualityError(f"Too many missing values for {symbol}: {missing_ratio:.2%}")
        
        logger.debug(f"Data quality validation passed for {symbol}")
    
    def _clean_stock_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and process stock data
        
        Args:
            data: Raw stock data
            symbol: Stock symbol
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        clean_data = data.copy()
        
        # Clean column names
        clean_data.columns = [col.lower().replace(" ", "_") for col in clean_data.columns]
        
        # Add symbol column
        clean_data['symbol'] = symbol
        
        # Calculate returns
        clean_data['returns'] = clean_data['close'].pct_change()
        clean_data['log_returns'] = np.log(clean_data['close'] / clean_data['close'].shift(1))
        
        # Calculate additional metrics
        clean_data['price_range'] = clean_data['high'] - clean_data['low']
        clean_data['price_change'] = clean_data['close'] - clean_data['open']
        clean_data['volume_ma_20'] = clean_data['volume'].rolling(20).mean()
        
        # Handle missing values
        numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
        clean_data[numeric_columns] = clean_data[numeric_columns].fillna(method='bfill').fillna(method='ffill')
        
        # Remove extreme outliers (beyond 5 standard deviations)
        for col in ['returns', 'log_returns']:
            if col in clean_data.columns:
                mean = clean_data[col].mean()
                std = clean_data[col].std()
                outlier_mask = np.abs(clean_data[col] - mean) > 5 * std
                clean_data.loc[outlier_mask, col] = np.nan
        
        return clean_data
    
    def fetch_all_stocks_data(self, 
                             period: str = "5y", 
                             interval: str = "1d",
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all SENSEX stocks with parallel processing
        
        Args:
            period: Data period
            interval: Data interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info(f"Starting to fetch data for {len(self.symbols)} stocks + index...")
        start_time = time.time()
        
        # Reset failed tickers list
        self.failed_tickers = []
        
        # Combine all symbols (index + stocks)
        all_symbols = [self.index_symbol] + self.symbols
        
        if parallel and len(all_symbols) > 1:
            all_data = self._fetch_parallel(all_symbols, period, interval, start_date, end_date)
        else:
            all_data = self._fetch_sequential(all_symbols, period, interval, start_date, end_date)
        
        # Cache the data
        self.data_cache = all_data
        
        # Log summary
        elapsed_time = time.time() - start_time
        success_count = len(all_data)
        total_count = len(all_symbols)
        
        logger.info("="*60)
        logger.info("DATA FETCHING SUMMARY")
        logger.info("="*60)
        logger.info(f"✓ Successfully fetched: {success_count}/{total_count} symbols")
        logger.info(f"✗ Failed symbols: {len(self.failed_tickers)}")
        logger.info(f"⏱  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"📊 Average time per symbol: {elapsed_time/total_count:.2f} seconds")
        
        if self.failed_tickers:
            logger.warning(f"Failed tickers: {', '.join(self.failed_tickers)}")
        
        return all_data
    
    def _fetch_parallel(self, symbols: List[str], period: str, interval: str, 
                       start_date: Optional[str], end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data using parallel processing"""
        logger.info(f"Using parallel fetching with {self.max_workers} workers")
        all_data = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_stock_data, symbol, period, interval, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        all_data[symbol] = data
                        logger.debug(f"✓ Completed {symbol}")
                    else:
                        logger.warning(f"✗ No data for {symbol}")
                except Exception as e:
                    logger.error(f"✗ Exception for {symbol}: {str(e)}")
        
        return all_data
    
    def _fetch_sequential(self, symbols: List[str], period: str, interval: str,
                         start_date: Optional[str], end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data sequentially"""
        logger.info("Using sequential fetching")
        all_data = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Fetching {symbol} ({i}/{len(symbols)})")
            
            data = self.fetch_stock_data(symbol, period, interval, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        return all_data
    
    def get_data_quality_report(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate a comprehensive data quality report
        
        Args:
            data: Dictionary of stock DataFrames
            
        Returns:
            Dictionary containing quality metrics
        """
        if not data:
            return {"status": "no_data", "details": "No data to analyze"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(data),
            "symbols_analyzed": list(data.keys()),
            "failed_symbols": self.failed_tickers.copy(),
            "quality_metrics": {}
        }
        
        # Analyze each symbol
        for symbol, df in data.items():
            metrics = {
                "record_count": len(df),
                "date_range": {
                    "start": df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                    "end": df.index.max().strftime('%Y-%m-%d') if not df.empty else None
                },
                "missing_values": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                "price_statistics": {
                    "avg_close": df['close'].mean() if 'close' in df.columns else None,
                    "min_close": df['close'].min() if 'close' in df.columns else None,
                    "max_close": df['close'].max() if 'close' in df.columns else None,
                    "volatility": df['returns'].std() if 'returns' in df.columns else None
                }
            }
            report["quality_metrics"][symbol] = metrics
        
        # Overall statistics
        total_records = sum(len(df) for df in data.values())
        total_missing = sum(df.isnull().sum().sum() for df in data.values())
        
        report["overall_statistics"] = {
            "total_records": total_records,
            "total_missing_values": total_missing,
            "overall_missing_percentage": (total_missing / (total_records * len(next(iter(data.values())).columns))) * 100 if data else 0,
            "average_records_per_symbol": total_records / len(data) if data else 0
        }
        
        return report
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        Save raw data to CSV files
        
        Args:
            data: Dictionary of DataFrames
            output_dir: Output directory path
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for symbol, df in data.items():
            if not df.empty:
                # Clean symbol name for filename
                clean_symbol = symbol.replace("^", "").replace(".NS", "")
                filename = f"{clean_symbol}_raw_data.csv"
                filepath = os.path.join(output_dir, filename)
                
                df.to_csv(filepath, index=True)
                logger.info(f"Saved {symbol} data to {filepath}")
    
    def load_raw_data(self, input_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from CSV files
        
        Args:
            input_dir: Input directory path
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return data
            
        for filename in os.listdir(input_dir):
            if filename.endswith('_raw_data.csv'):
                filepath = os.path.join(input_dir, filename)
                
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    
                    # Extract symbol from filename or DataFrame
                    if 'symbol' in df.columns:
                        symbol = df['symbol'].iloc[0]
                    else:
                        symbol = filename.replace('_raw_data.csv', '')
                        if symbol == 'BSESN':
                            symbol = '^BSESN'
                        else:
                            symbol = f"{symbol}.NS"
                    
                    data[symbol] = df
                    logger.info(f"Loaded data for {symbol} from {filepath}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {str(e)}")
                    
        return data
    
    def get_date_range(self, data: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
        """
        Get the common date range across all stocks
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if not data:
            return None, None
            
        start_dates = []
        end_dates = []
        
        for df in data.values():
            if not df.empty:
                start_dates.append(df.index.min())
                end_dates.append(df.index.max())
        
        if start_dates and end_dates:
            return max(start_dates), min(end_dates)
        else:
            return None, None
    
    def align_data_by_date(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all DataFrames to have the same date range
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dictionary of aligned DataFrames
        """
        start_date, end_date = self.get_date_range(data)
        
        if start_date is None or end_date is None:
            logger.warning("Could not determine common date range")
            return data
            
        aligned_data = {}
        
        for symbol, df in data.items():
            if not df.empty:
                # Filter to common date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                aligned_df = df.loc[mask].copy()
                
                if not aligned_df.empty:
                    aligned_data[symbol] = aligned_df
                    
        logger.info(f"Aligned data to date range: {start_date} to {end_date}")
        logger.info(f"Common data points: {len(aligned_data[list(aligned_data.keys())[0]])}")
        
        return aligned_data


def main():
    """
    Main function to fetch and save SENSEX data
    """
    # Import configuration
    from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX, DATA_CONFIG, PATHS
    
    # Initialize data fetcher
    fetcher = SensexDataFetcher(SENSEX_30_SYMBOLS, SENSEX_INDEX)
    
    # Fetch all data
    all_data = fetcher.fetch_all_stocks_data(
        period=DATA_CONFIG['period'],
        interval=DATA_CONFIG['interval']
    )
    
    # Align data by common date range
    aligned_data = fetcher.align_data_by_date(all_data)
    
    # Save raw data
    fetcher.save_raw_data(aligned_data, PATHS['raw_data'])
    
    logger.info("Data fetching completed successfully!")


if __name__ == "__main__":
    main()
