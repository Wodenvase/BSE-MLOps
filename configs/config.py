# SENSEX 30 component stocks (as of 2024)
SENSEX_30_SYMBOLS = [
    "RELIANCE.NS",    # Reliance Industries
    "HDFCBANK.NS",    # HDFC Bank
    "TCS.NS",         # Tata Consultancy Services
    "INFY.NS",        # Infosys
    "ICICIBANK.NS",   # ICICI Bank
    "HINDUNILVR.NS",  # Hindustan Unilever
    "SBIN.NS",        # State Bank of India
    "BHARTIARTL.NS",  # Bharti Airtel
    "ITC.NS",         # ITC
    "ASIANPAINT.NS",  # Asian Paints
    "AXISBANK.NS",    # Axis Bank
    "LT.NS",          # Larsen & Toubro
    "HCLTECH.NS",     # HCL Technologies
    "WIPRO.NS",       # Wipro
    "MARUTI.NS",      # Maruti Suzuki
    "SUNPHARMA.NS",   # Sun Pharmaceutical
    "POWERGRID.NS",   # Power Grid Corporation
    "NTPC.NS",        # NTPC
    "ULTRACEMCO.NS",  # UltraTech Cement
    "ONGC.NS",        # Oil and Natural Gas Corporation
    "TECHM.NS",       # Tech Mahindra
    "KOTAKBANK.NS",   # Kotak Mahindra Bank
    "M&M.NS",         # Mahindra & Mahindra
    "TITAN.NS",       # Titan Company
    "INDUSINDBK.NS",  # IndusInd Bank
    "BAJFINANCE.NS",  # Bajaj Finance
    "NESTLEIND.NS",   # Nestle India
    "HDFCLIFE.NS",    # HDFC Life Insurance
    "BAJAJFINSV.NS",  # Bajaj Finserv
    "DRREDDY.NS"      # Dr. Reddy's Laboratories
]

# SENSEX index symbol
SENSEX_INDEX = "^BSESN"

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'RSI_14': {'period': 14},
    'RSI_21': {'period': 21},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'SMA_20': {'period': 20},
    'SMA_50': {'period': 50},
    'EMA_12': {'period': 12},
    'EMA_26': {'period': 26},
    'BBANDS_20': {'period': 20, 'std': 2},
    'STOCH_K': {'k_period': 14, 'd_period': 3},
    'STOCH_D': {'k_period': 14, 'd_period': 3},
    'ADX_14': {'period': 14},
    'CCI_20': {'period': 20},
    'WILLR_14': {'period': 14},
    'ROC_10': {'period': 10},
    'MOMENTUM_10': {'period': 10}
}

# Data fetch configuration
DATA_CONFIG = {
    'period': '5y',  # 5 years of historical data
    'interval': '1d',  # Daily data
    'start_date': None,  # Will be calculated dynamically
    'end_date': None,    # Will be calculated dynamically
}

# Model configuration
MODEL_CONFIG = {
    'sequence_length': 30,  # 30 days of historical data
    'n_features': len(TECHNICAL_INDICATORS) + 4,  # Technical indicators + OHLC returns
    'n_stocks': len(SENSEX_30_SYMBOLS),  # 30 stocks
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'test_split': 0.1
}

# MLflow configuration
MLFLOW_CONFIG = {
    'experiment_name': 'sensex_convlstm_forecasting',
    'tracking_uri': 'http://localhost:5000',
    'artifact_location': './mlruns'
}

# Paths
PATHS = {
    'raw_data': './data/raw',
    'processed_data': './data/processed', 
    'models': './models',
    'logs': './logs'
}
