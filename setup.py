"""
Setup script for initial project configuration
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/processed/prediction",
        "models",
        "logs",
        "mlruns",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def initialize_dvc():
    """Initialize DVC"""
    try:
        if not os.path.exists('.dvc'):
            subprocess.check_call(["dvc", "init"])
            logger.info("DVC initialized")
        else:
            logger.info("DVC already initialized")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error initializing DVC: {e}")
        return False
    except FileNotFoundError:
        logger.warning("DVC not found. Please install with: pip install dvc[gdrive]")
        return False
    return True

def initialize_git():
    """Initialize Git repository"""
    try:
        if not os.path.exists('.git'):
            subprocess.check_call(["git", "init"])
            subprocess.check_call(["git", "add", "."])
            subprocess.check_call(["git", "commit", "-m", "Initial commit"])
            logger.info("Git repository initialized")
        else:
            logger.info("Git repository already exists")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error initializing Git: {e}")
        return False
    except FileNotFoundError:
        logger.warning("Git not found. Please install Git")
        return False
    return True

def create_env_file():
    """Create .env file with default values"""
    env_content = '''# MLOps Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_USERNAME=admin
MLFLOW_PASSWORD=password123

# Airflow Configuration  
AIRFLOW_UID=50000
AIRFLOW_GID=0
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123

# Database Configuration
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Model Configuration
MODEL_SEQUENCE_LENGTH=30
MODEL_N_FEATURES=24
MODEL_N_STOCKS=30

# API Keys (to be filled by user)
# GDRIVE_FOLDER_ID=your_google_drive_folder_id
# HF_TOKEN=your_hugging_face_token
# HF_USERNAME=your_hugging_face_username
# SLACK_WEBHOOK=your_slack_webhook_url
'''
    
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info("Created .env file with default values")
    else:
        logger.info(".env file already exists")

def setup_pre_commit_hooks():
    """Setup pre-commit hooks for code quality"""
    pre_commit_config = '''repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
  
  - repo: local
    hooks:
      - id: dvc-check
        name: DVC check
        entry: dvc status
        language: system
        always_run: true
        pass_filenames: false
'''
    
    pre_commit_file = Path('.pre-commit-config.yaml')
    if not pre_commit_file.exists():
        with open(pre_commit_file, 'w') as f:
            f.write(pre_commit_config)
        logger.info("Created pre-commit configuration")
        
        # Install pre-commit hooks
        try:
            subprocess.check_call(["pre-commit", "install"])
            logger.info("Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not install pre-commit hooks. Install with: pip install pre-commit")

def create_jupyter_notebook():
    """Create a sample Jupyter notebook for exploration"""
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SENSEX ConvLSTM Forecasting - Data Exploration\\n",
    "\\n",
    "This notebook provides data exploration and model analysis for the SENSEX forecasting project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import sys\\n",
    "\\n",
    "# Add src to path\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "from data.fetch_data import SensexDataFetcher\\n",
    "from data.create_feature_maps import FeatureEngineer\\n",
    "from configs.config import SENSEX_30_SYMBOLS, SENSEX_INDEX\\n",
    "\\n",
    "plt.style.use('default')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Basic Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Fetch sample data\\n",
    "fetcher = SensexDataFetcher(SENSEX_30_SYMBOLS[:5], SENSEX_INDEX)\\n",
    "data = fetcher.fetch_all_stocks_data(period='1y')\\n",
    "\\n",
    "print(f'Fetched data for {len(data)} symbols')\\n",
    "print(f'Date range: {min([df.index.min() for df in data.values()])} to {max([df.index.max() for df in data.values()])}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create feature maps\\n",
    "engineer = FeatureEngineer(SENSEX_30_SYMBOLS[:5])\\n",
    "feature_maps, targets, dates = engineer.create_component_feature_maps(data, SENSEX_INDEX)\\n",
    "\\n",
    "print(f'Feature maps shape: {feature_maps.shape}')\\n",
    "print(f'Targets shape: {targets.shape}')\\n",
    "print(f'Feature names: {engineer.feature_names}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Plot SENSEX index\\n",
    "sensex_data = data[SENSEX_INDEX]\\n",
    "\\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\\n",
    "\\n",
    "# Price chart\\n",
    "axes[0,0].plot(sensex_data.index, sensex_data['close'])\\n",
    "axes[0,0].set_title('SENSEX Close Price')\\n",
    "axes[0,0].set_ylabel('Price')\\n",
    "\\n",
    "# Returns distribution\\n",
    "axes[0,1].hist(sensex_data['returns'].dropna(), bins=50, alpha=0.7)\\n",
    "axes[0,1].set_title('Returns Distribution')\\n",
    "axes[0,1].set_xlabel('Daily Returns')\\n",
    "\\n",
    "# Volume\\n",
    "axes[1,0].plot(sensex_data.index, sensex_data['volume'])\\n",
    "axes[1,0].set_title('Trading Volume')\\n",
    "axes[1,0].set_ylabel('Volume')\\n",
    "\\n",
    "# Target distribution\\n",
    "axes[1,1].bar(['Down', 'Up'], [np.sum(targets == 0), np.sum(targets == 1)])\\n",
    "axes[1,1].set_title('Target Class Distribution')\\n",
    "axes[1,1].set_ylabel('Count')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze feature correlations for the first stock\\n",
    "stock_features = feature_maps[:, 0, :]  # First stock\\n",
    "feature_df = pd.DataFrame(stock_features, columns=engineer.feature_names)\\n",
    "\\n",
    "# Correlation matrix\\n",
    "corr_matrix = feature_df.corr()\\n",
    "\\n",
    "plt.figure(figsize=(12, 10))\\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')\\n",
    "plt.title('Feature Correlation Matrix')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    notebook_file = Path('notebooks/data_exploration.ipynb')
    if not notebook_file.exists():
        with open(notebook_file, 'w') as f:
            f.write(notebook_content)
        logger.info("Created sample Jupyter notebook")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SENSEX MLOps Project Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Configure API Keys and Credentials:")
    print("   - Edit .env file with your API keys")
    print("   - Setup Google Drive folder for DVC remote storage")
    print("   - Get Hugging Face token for deployment")
    
    print("\n2. Start Local Development:")
    print("   - Run: docker-compose up -d")
    print("   - Access services:")
    print("     • Airflow: http://localhost:8080 (admin/admin123)")
    print("     • MLflow: http://localhost:5000") 
    print("     • Streamlit: http://localhost:8501")
    print("     • Jupyter: http://localhost:8888")
    
    print("\n3. Run the Pipeline:")
    print("   - Fetch data: python src/data/fetch_data.py")
    print("   - Create features: python src/data/create_feature_maps.py")
    print("   - Train model: python src/models/convlstm_model.py")
    
    print("\n4. Setup Version Control:")
    print("   - Initialize DVC remote: python src/utils/dvc_utils.py")
    print("   - Push to GitHub and setup Actions secrets")
    
    print("\n5. Deploy to Production:")
    print("   - Configure Hugging Face Spaces deployment")
    print("   - Monitor via GitHub Actions")
    
    print("\n📚 Documentation:")
    print("   - Check README.md for detailed instructions") 
    print("   - Explore notebooks/ for data analysis")
    print("   - Review configs/ for configuration options")
    
    print("\n⚠️  Important Notes:")
    print("   - This is for educational purposes only")
    print("   - Not financial advice - consult professionals")
    print("   - Test thoroughly before production use")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    logger.info("Starting SENSEX MLOps project setup...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return
    
    # Initialize version control
    initialize_git()
    initialize_dvc()
    
    # Create configuration files
    create_env_file()
    setup_pre_commit_hooks()
    
    # Create sample notebook
    create_jupyter_notebook()
    
    # Print next steps
    print_next_steps()
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    main()
