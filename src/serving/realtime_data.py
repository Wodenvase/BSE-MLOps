"""
Real-time Data Pipeline for SENSEX Prediction App
Fetches latest market data for live predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to path
sys.path.append('../')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataFetcher:
    """
    Real-time data fetcher for SENSEX components and index
    Optimized for Streamlit application with caching and error handling
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize Real-time Data Fetcher
        
        Args:
            symbols: List of stock symbols to fetch
        """
        # Default SENSEX 30 symbols
        self.default_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'ASIANPAINT.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS', 'WIPRO.NS',
            'POWERGRID.NS', 'NTPC.NS', 'TATASTEEL.NS', 'HCLTECH.NS', 'COALINDIA.NS',
            'ONGC.NS', 'GRASIM.NS', 'TECHM.NS', 'DRREDDY.NS', 'JSWSTEEL.NS'
        ]
        
        self.symbols = symbols if symbols else self.default_symbols
        self.sensex_symbol = '^BSESN'
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        logger.info(f"Initialized RealTimeDataFetcher with {len(self.symbols)} symbols")
    
    def fetch_latest_data(self, period: str = "60d", 
                         use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest market data for all symbols
        
        Args:
            period: Data period (default 60d for sufficient sequence length)
            use_cache: Whether to use cached data
            
        Returns:
            Dict: Symbol -> DataFrame mapping
        """
        try:
            cache_key = f"latest_data_{period}"
            
            # Check cache first
            if use_cache and self._is_cache_valid(cache_key):
                logger.info("Using cached data")
                return self.cache[cache_key]['data']
            
            logger.info(f"Fetching latest data for {len(self.symbols)} symbols...")
            
            # Fetch data in parallel
            all_data = {}
            
            # Add SENSEX index to symbols
            all_symbols = self.symbols + [self.sensex_symbol]
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all download tasks
                future_to_symbol = {
                    executor.submit(self._fetch_single_stock, symbol, period): symbol 
                    for symbol in all_symbols
                }
                
                # Collect results
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result(timeout=30)
                        if data is not None and not data.empty:
                            all_data[symbol] = data
                            logger.info(f"✅ {symbol}: {len(data)} rows")
                        else:
                            logger.warning(f"❌ {symbol}: No data")
                    except Exception as e:
                        logger.error(f"❌ {symbol}: {str(e)}")
            
            # Cache results
            if all_data:
                self.cache[cache_key] = {
                    'data': all_data,
                    'timestamp': datetime.now()
                }
                logger.info(f"Cached data for {len(all_data)} symbols")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            return {}
    
    def _fetch_single_stock(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single stock with retry logic
        
        Args:
            symbol: Stock symbol
            period: Data period
            
        Returns:
            pd.DataFrame: Stock data or None
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, auto_adjust=True, prepost=True)
                
                if data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return None
                
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return None
                
                # Remove any rows with NaN values in critical columns
                data = data.dropna(subset=['close'])
                
                if len(data) < 30:  # Need at least 30 days for sequence
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return None
                
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All attempts failed for {symbol}: {str(e)}")
                    return None
        
        return None
    
    def get_latest_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Get latest prices for all symbols (optimized for quick updates)
        
        Returns:
            Dict: Symbol -> price info mapping
        """
        try:
            cache_key = "latest_prices"
            
            # Check cache (shorter TTL for prices)
            if self._is_cache_valid(cache_key, ttl=60):  # 1 minute cache
                return self.cache[cache_key]['data']
            
            logger.info("Fetching latest prices...")
            
            price_data = {}
            all_symbols = self.symbols + [self.sensex_symbol]
            
            # Batch fetch using yfinance
            try:
                tickers = yf.Tickers(' '.join(all_symbols))
                
                for symbol in all_symbols:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
                        
                        if current_price and prev_close:
                            change = current_price - prev_close
                            change_pct = (change / prev_close) * 100
                            
                            price_data[symbol] = {
                                'current_price': float(current_price),
                                'previous_close': float(prev_close),
                                'change': float(change),
                                'change_percent': float(change_pct),
                                'timestamp': datetime.now().isoformat()
                            }
                        
                    except Exception as e:
                        logger.warning(f"Error getting price for {symbol}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in batch price fetch: {str(e)}")
                # Fallback to individual fetches
                return self._fetch_prices_individually(all_symbols)
            
            # Cache results
            if price_data:
                self.cache[cache_key] = {
                    'data': price_data,
                    'timestamp': datetime.now()
                }
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching latest prices: {str(e)}")
            return {}
    
    def _fetch_prices_individually(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fallback method to fetch prices individually"""
        price_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if len(hist) >= 2:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2])
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    price_data[symbol] = {
                        'current_price': current_price,
                        'previous_close': prev_close,
                        'change': change,
                        'change_percent': change_pct,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.warning(f"Individual fetch failed for {symbol}: {str(e)}")
                continue
        
        return price_data
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get market summary including SENSEX performance and top movers
        
        Returns:
            Dict: Market summary data
        """
        try:
            cache_key = "market_summary"
            
            if self._is_cache_valid(cache_key, ttl=300):  # 5 minute cache
                return self.cache[cache_key]['data']
            
            logger.info("Generating market summary...")
            
            # Get latest prices
            prices = self.get_latest_prices()
            
            if not prices:
                return {}
            
            # Extract SENSEX data
            sensex_data = prices.get(self.sensex_symbol, {})
            
            # Find top gainers and losers
            stock_changes = []
            for symbol, data in prices.items():
                if symbol != self.sensex_symbol and 'change_percent' in data:
                    stock_changes.append({
                        'symbol': symbol.replace('.NS', ''),
                        'change_percent': data['change_percent'],
                        'current_price': data['current_price']
                    })
            
            # Sort by performance
            stock_changes.sort(key=lambda x: x['change_percent'], reverse=True)
            
            top_gainers = stock_changes[:5]
            top_losers = stock_changes[-5:]
            
            # Calculate market breadth
            advancing = len([s for s in stock_changes if s['change_percent'] > 0])
            declining = len([s for s in stock_changes if s['change_percent'] < 0])
            unchanged = len([s for s in stock_changes if s['change_percent'] == 0])
            
            summary = {
                'sensex': sensex_data,
                'top_gainers': top_gainers,
                'top_losers': top_losers,
                'market_breadth': {
                    'advancing': advancing,
                    'declining': declining,
                    'unchanged': unchanged,
                    'advance_decline_ratio': advancing / declining if declining > 0 else 0
                },
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(stock_changes)
            }
            
            # Cache results
            self.cache[cache_key] = {
                'data': summary,
                'timestamp': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating market summary: {str(e)}")
            return {}
    
    def _is_cache_valid(self, key: str, ttl: int = None) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            key: Cache key
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            bool: True if cache is valid
        """
        if key not in self.cache:
            return False
        
        cache_ttl = ttl if ttl is not None else self.cache_ttl
        elapsed = (datetime.now() - self.cache[key]['timestamp']).total_seconds()
        
        return elapsed < cache_ttl
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_ages': {}
        }
        
        for key, data in self.cache.items():
            age = (datetime.now() - data['timestamp']).total_seconds()
            stats['cache_ages'][key] = age
        
        return stats
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate data quality for prediction readiness
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dict: Data quality report
        """
        try:
            report = {
                'status': 'valid',
                'issues': [],
                'warnings': [],
                'statistics': {},
                'readiness': True
            }
            
            required_symbols = 25  # Need at least 25 out of 30 symbols
            min_days = 30  # Need at least 30 days of data
            
            # Check symbol availability
            valid_symbols = 0
            for symbol, df in data.items():
                if symbol == self.sensex_symbol:
                    continue
                    
                if df is not None and not df.empty and len(df) >= min_days:
                    valid_symbols += 1
                else:
                    report['issues'].append(f"Insufficient data for {symbol}")
            
            if valid_symbols < required_symbols:
                report['status'] = 'invalid'
                report['readiness'] = False
                report['issues'].append(f"Only {valid_symbols}/{required_symbols} symbols have sufficient data")
            
            # Check data recency
            if data:
                latest_dates = []
                for symbol, df in data.items():
                    if df is not None and not df.empty:
                        latest_dates.append(df.index[-1])
                
                if latest_dates:
                    most_recent = max(latest_dates)
                    age_hours = (datetime.now() - most_recent.to_pydatetime()).total_seconds() / 3600
                    
                    if age_hours > 24:
                        report['warnings'].append(f"Data is {age_hours:.1f} hours old")
                        if age_hours > 72:
                            report['status'] = 'stale'
                            report['readiness'] = False
            
            # Statistics
            report['statistics'] = {
                'valid_symbols': valid_symbols,
                'total_symbols': len(self.symbols),
                'data_coverage': valid_symbols / len(self.symbols) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'readiness': False
            }

def main():
    """
    Test the Real-time Data Fetcher
    """
    print("🚀 Testing Real-time Data Fetcher...")
    
    fetcher = RealTimeDataFetcher()
    
    # Test latest prices
    print("💰 Fetching latest prices...")
    prices = fetcher.get_latest_prices()
    print(f"✅ Got prices for {len(prices)} symbols")
    
    # Test market summary
    print("📊 Generating market summary...")
    summary = fetcher.get_market_summary()
    if summary:
        print(f"📈 SENSEX: {summary.get('sensex', {}).get('current_price', 'N/A')}")
        print(f"🎯 Top Gainer: {summary.get('top_gainers', [{}])[0].get('symbol', 'N/A')}")
    
    # Test data fetching (smaller sample)
    print("📥 Fetching sample data...")
    sample_symbols = fetcher.default_symbols[:5]  # Test with 5 symbols
    fetcher.symbols = sample_symbols
    
    data = fetcher.fetch_latest_data(period="60d")
    print(f"✅ Got data for {len(data)} symbols")
    
    # Validate data quality
    print("🔍 Validating data quality...")
    quality = fetcher.validate_data_quality(data)
    print(f"📋 Data Status: {quality['status']}")
    print(f"🎯 Prediction Ready: {quality['readiness']}")
    
    # Cache stats
    cache_stats = fetcher.get_cache_stats()
    print(f"💾 Cache: {cache_stats['cache_size']} entries")

if __name__ == "__main__":
    main()
