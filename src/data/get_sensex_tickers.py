"""
SENSEX Component Ticker Scraper
Fetches the current list of 30 SENSEX component stocks from multiple reliable sources.
"""

import requests
import pandas as pd
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from pathlib import Path
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SensexTickerScraper:
    """
    Scrapes SENSEX 30 component tickers from multiple reliable sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Fallback list - current SENSEX 30 components (as of 2024)
        self.fallback_tickers = [
            "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "ASIANPAINT.NS",
            "AXISBANK.NS", "LT.NS", "HCLTECH.NS", "WIPRO.NS", "MARUTI.NS",
            "SUNPHARMA.NS", "POWERGRID.NS", "NTPC.NS", "ULTRACEMCO.NS", "ONGC.NS",
            "TECHM.NS", "KOTAKBANK.NS", "M&M.NS", "TITAN.NS", "INDUSINDBK.NS",
            "BAJFINANCE.NS", "NESTLEIND.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "DRREDDY.NS"
        ]
    
    def scrape_from_bse_website(self) -> Optional[List[str]]:
        """
        Scrape SENSEX components from BSE official website
        """
        try:
            logger.info("Attempting to scrape from BSE website...")
            
            # BSE SENSEX constituents page
            url = "https://www.bseindia.com/indices/IndexArchiveData.html"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for SENSEX components table
                # This is a simplified approach - actual implementation would need
                # to parse the specific table structure on BSE website
                
                logger.warning("BSE website scraping needs specific implementation")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping BSE website: {str(e)}")
            return None
    
    def scrape_from_moneycontrol(self) -> Optional[List[str]]:
        """
        Scrape SENSEX components from MoneyControl
        """
        try:
            logger.info("Attempting to scrape from MoneyControl...")
            
            url = "https://www.moneycontrol.com/indian-indices/sensex-4.html"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for stock symbols in the page
                # This would need specific parsing logic for MoneyControl's structure
                
                logger.warning("MoneyControl scraping needs specific implementation")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping MoneyControl: {str(e)}")
            return None
    
    def scrape_from_investing_com(self) -> Optional[List[str]]:
        """
        Scrape SENSEX components from Investing.com
        """
        try:
            logger.info("Attempting to scrape from Investing.com...")
            
            url = "https://www.investing.com/indices/sensex-components"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for component stocks table
                tickers = []
                
                # Find table rows with stock data
                rows = soup.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) > 0:
                        # Look for stock symbols - this would need refinement
                        for cell in cells:
                            text = cell.get_text().strip()
                            # Simple heuristic - if it looks like a stock symbol
                            if text and len(text) < 15 and text.isupper():
                                if not text.endswith('.NS'):
                                    text += '.NS'
                                if text not in tickers and self._is_valid_ticker(text):
                                    tickers.append(text)
                
                if len(tickers) >= 25:  # Should have close to 30 components
                    logger.info(f"Found {len(tickers)} tickers from Investing.com")
                    return tickers[:30]  # Take first 30
                
        except Exception as e:
            logger.error(f"Error scraping Investing.com: {str(e)}")
            
        return None
    
    def get_tickers_from_yfinance_index(self) -> Optional[List[str]]:
        """
        Try to get SENSEX components using yfinance index data
        """
        try:
            logger.info("Attempting to get tickers from yfinance...")
            
            # Get SENSEX index info
            sensex = yf.Ticker("^BSESN")
            
            # Unfortunately, yfinance doesn't directly provide index components
            # This would need a different approach or API
            
            logger.warning("yfinance doesn't provide index components directly")
            return None
            
        except Exception as e:
            logger.error(f"Error getting tickers from yfinance: {str(e)}")
            return None
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol is reasonable
        """
        if not ticker or len(ticker) < 3:
            return False
        
        # Should end with .NS for NSE stocks
        if not ticker.endswith('.NS'):
            return False
        
        # Should have reasonable length
        symbol_part = ticker.replace('.NS', '')
        if len(symbol_part) < 2 or len(symbol_part) > 15:
            return False
        
        return True
    
    def validate_tickers_with_yfinance(self, tickers: List[str]) -> List[str]:
        """
        Validate tickers by checking if they exist in yfinance
        """
        logger.info("Validating tickers with yfinance...")
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Try to get basic info for the ticker
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if we got valid data
                if info and 'symbol' in info:
                    valid_tickers.append(ticker)
                    logger.debug(f"✓ {ticker} is valid")
                else:
                    logger.warning(f"✗ {ticker} may not be valid")
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"✗ {ticker} validation failed: {str(e)}")
                
        logger.info(f"Validated {len(valid_tickers)}/{len(tickers)} tickers")
        return valid_tickers
    
    def get_sensex_components(self, validate: bool = True) -> List[str]:
        """
        Get SENSEX 30 component tickers from multiple sources
        
        Args:
            validate: Whether to validate tickers with yfinance
            
        Returns:
            List of ticker symbols
        """
        logger.info("Starting SENSEX component ticker scraping...")
        
        # Try multiple sources
        sources = [
            self.scrape_from_investing_com,
            self.scrape_from_moneycontrol,
            self.scrape_from_bse_website,
            self.get_tickers_from_yfinance_index
        ]
        
        for source_func in sources:
            try:
                tickers = source_func()
                if tickers and len(tickers) >= 25:  # Should have most components
                    logger.info(f"Successfully got {len(tickers)} tickers from {source_func.__name__}")
                    
                    if validate:
                        tickers = self.validate_tickers_with_yfinance(tickers)
                    
                    if len(tickers) >= 25:
                        return tickers[:30]  # Return first 30
                    
            except Exception as e:
                logger.error(f"Error with {source_func.__name__}: {str(e)}")
                continue
        
        # If all sources fail, use fallback list
        logger.warning("All scraping sources failed, using fallback ticker list")
        
        if validate:
            logger.info("Validating fallback tickers...")
            valid_fallback = self.validate_tickers_with_yfinance(self.fallback_tickers)
            if len(valid_fallback) >= 25:
                return valid_fallback
        
        return self.fallback_tickers
    
    def save_tickers_to_file(self, tickers: List[str], output_path: str = "data/sensex_components.json"):
        """
        Save tickers to a JSON file with metadata
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "tickers": tickers,
            "count": len(tickers),
            "last_updated": datetime.now().isoformat(),
            "source": "automated_scraper"
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(tickers)} tickers to {output_path}")
    
    def load_tickers_from_file(self, file_path: str = "data/sensex_components.json") -> Optional[List[str]]:
        """
        Load tickers from a previously saved file
        """
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                tickers = data.get('tickers', [])
                last_updated = data.get('last_updated')
                
                logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
                logger.info(f"Last updated: {last_updated}")
                
                return tickers
            
        except Exception as e:
            logger.error(f"Error loading tickers from file: {str(e)}")
        
        return None


def main():
    """
    Main function to run the ticker scraping
    """
    scraper = SensexTickerScraper()
    
    # Try to load from file first (for efficiency)
    tickers = scraper.load_tickers_from_file()
    
    # If no file or data is old, scrape fresh data
    if not tickers:
        logger.info("No cached tickers found, scraping fresh data...")
        tickers = scraper.get_sensex_components(validate=True)
        
        # Save to file for future use
        scraper.save_tickers_to_file(tickers)
    
    # Display results
    logger.info("="*50)
    logger.info("SENSEX 30 COMPONENT TICKERS:")
    logger.info("="*50)
    
    for i, ticker in enumerate(tickers, 1):
        company_name = ticker.replace('.NS', '')
        logger.info(f"{i:2d}. {ticker:15s} ({company_name})")
    
    logger.info("="*50)
    logger.info(f"Total: {len(tickers)} components")
    
    return tickers


if __name__ == "__main__":
    main()
