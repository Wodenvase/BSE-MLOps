"""
Deployment script for Hugging Face Spaces
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, create_repo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_to_hf_spaces():
    """
    Deploy the Streamlit app to Hugging Face Spaces
    """
    # Get environment variables
    hf_token = os.getenv('HF_TOKEN')
    hf_username = os.getenv('HF_USERNAME')
    repo_name = 'sensex-convlstm-forecasting'
    
    if not hf_token or not hf_username:
        logger.error("HF_TOKEN and HF_USERNAME must be set")
        return
    
    # Initialize HF API
    api = HfApi(token=hf_token)
    
    # Create repository if it doesn't exist
    repo_id = f"{hf_username}/{repo_name}"
    
    try:
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True
        )
        logger.info(f"Repository {repo_id} created/updated")
    except Exception as e:
        logger.error(f"Error creating repository: {str(e)}")
        return
    
    # Prepare files for deployment
    files_to_upload = []
    
    # Copy Streamlit app
    if os.path.exists('streamlit_app/app.py'):
        shutil.copy('streamlit_app/app.py', 'app.py')
        files_to_upload.append('app.py')
    
    # Create requirements.txt for HF Spaces
    hf_requirements = [
        'streamlit==1.25.0',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'plotly==5.15.0',
        'yfinance==0.2.18',
        'ta==0.10.2',
        'scikit-learn==1.3.0'
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(hf_requirements))
    files_to_upload.append('requirements.txt')
    
    # Create README for HF Spaces
    readme_content = """---
title: SENSEX ConvLSTM Forecasting
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
license: mit
---

# SENSEX ConvLSTM Forecasting Dashboard

An advanced deep learning application for forecasting SENSEX index movement using ConvLSTM neural networks and Component Feature Maps.

## Features

- Real-time SENSEX data visualization
- ConvLSTM-based directional predictions
- Interactive technical analysis dashboard  
- Component stock analysis
- Model performance metrics

## Technology Stack

- **Deep Learning**: ConvLSTM (TensorFlow/Keras)
- **Data Source**: yfinance API
- **Features**: 24 technical indicators across 30 SENSEX components
- **Visualization**: Plotly, Streamlit
- **MLOps**: MLflow, DVC, Apache Airflow

## Disclaimer

This application is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment choices.

## Usage

The dashboard provides:
1. Latest market predictions (Up/Down)
2. Confidence scores and probability estimates
3. Interactive charts and visualizations
4. Historical performance analysis
5. Technical indicator analysis

Built with ❤️ using 100% free and open-source tools.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    files_to_upload.append('README.md')
    
    # Copy essential source files
    essential_dirs = ['src', 'configs']
    for dir_name in essential_dirs:
        if os.path.exists(dir_name):
            # Copy directory contents
            shutil.copytree(dir_name, f"./{dir_name}", dirs_exist_ok=True)
    
    # Upload files to HF Spaces
    try:
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="space",
                    token=hf_token
                )
                logger.info(f"Uploaded {file_path}")
        
        # Upload source directories
        for dir_name in essential_dirs:
            if os.path.exists(dir_name):
                api.upload_folder(
                    folder_path=dir_name,
                    repo_id=repo_id,
                    repo_type="space",
                    token=hf_token
                )
                logger.info(f"Uploaded directory {dir_name}")
        
        logger.info(f"Deployment to HF Spaces completed successfully!")
        logger.info(f"App URL: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        logger.error(f"Error uploading to HF Spaces: {str(e)}")
        return

if __name__ == "__main__":
    deploy_to_hf_spaces()
