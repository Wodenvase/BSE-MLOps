"""
DVC setup and utilities for data and model versioning
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DVCManager:
    """
    Manages DVC operations for data and model versioning
    """
    
    def __init__(self, project_root: str = "./"):
        """
        Initialize DVC manager
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        
    def init_dvc(self):
        """
        Initialize DVC in the project
        """
        try:
            if not self.dvc_dir.exists():
                result = subprocess.run(
                    ["dvc", "init"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("DVC initialized successfully")
                else:
                    logger.error(f"DVC initialization failed: {result.stderr}")
                    
            else:
                logger.info("DVC already initialized")
                
        except Exception as e:
            logger.error(f"Error initializing DVC: {str(e)}")
    
    def add_remote(self, remote_name: str, remote_url: str, set_default: bool = True):
        """
        Add remote storage for DVC
        
        Args:
            remote_name: Name of the remote
            remote_url: URL of the remote storage
            set_default: Whether to set as default remote
        """
        try:
            # Add remote
            result = subprocess.run(
                ["dvc", "remote", "add", remote_name, remote_url],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Added DVC remote: {remote_name}")
                
                # Set as default if requested
                if set_default:
                    self.set_default_remote(remote_name)
            else:
                logger.error(f"Failed to add remote: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error adding DVC remote: {str(e)}")
    
    def set_default_remote(self, remote_name: str):
        """
        Set default remote for DVC
        
        Args:
            remote_name: Name of the remote to set as default
        """
        try:
            result = subprocess.run(
                ["dvc", "remote", "default", remote_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Set default DVC remote: {remote_name}")
            else:
                logger.error(f"Failed to set default remote: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error setting default remote: {str(e)}")
    
    def add_data(self, data_path: str):
        """
        Add data file or directory to DVC tracking
        
        Args:
            data_path: Path to data file or directory
        """
        try:
            result = subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Added to DVC tracking: {data_path}")
                
                # Add .dvc file to git
                dvc_file = f"{data_path}.dvc"
                if os.path.exists(dvc_file):
                    subprocess.run(["git", "add", dvc_file], cwd=self.project_root)
                    
            else:
                logger.error(f"Failed to add to DVC: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error adding data to DVC: {str(e)}")
    
    def push_data(self, targets: Optional[List[str]] = None):
        """
        Push data to remote storage
        
        Args:
            targets: Specific targets to push (if None, pushes all)
        """
        try:
            cmd = ["dvc", "push"]
            if targets:
                cmd.extend(targets)
                
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Data pushed to remote storage successfully")
            else:
                logger.error(f"Failed to push data: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error pushing data: {str(e)}")
    
    def pull_data(self, targets: Optional[List[str]] = None):
        """
        Pull data from remote storage
        
        Args:
            targets: Specific targets to pull (if None, pulls all)
        """
        try:
            cmd = ["dvc", "pull"]
            if targets:
                cmd.extend(targets)
                
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Data pulled from remote storage successfully")
            else:
                logger.error(f"Failed to pull data: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error pulling data: {str(e)}")
    
    def create_pipeline(self, pipeline_config: Dict):
        """
        Create DVC pipeline
        
        Args:
            pipeline_config: Pipeline configuration dictionary
        """
        try:
            # Create dvc.yaml file
            dvc_yaml_path = self.project_root / "dvc.yaml"
            
            with open(dvc_yaml_path, 'w') as f:
                yaml.dump(pipeline_config, f, default_flow_style=False)
            
            logger.info("DVC pipeline created successfully")
            
        except Exception as e:
            logger.error(f"Error creating DVC pipeline: {str(e)}")
    
    def run_pipeline(self, stage: Optional[str] = None):
        """
        Run DVC pipeline
        
        Args:
            stage: Specific stage to run (if None, runs all)
        """
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
                
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("DVC pipeline executed successfully")
            else:
                logger.error(f"Pipeline execution failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
    
    def get_status(self):
        """
        Get DVC status
        """
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("DVC Status:\n" + result.stdout)
                return result.stdout
            else:
                logger.error(f"Failed to get status: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting DVC status: {str(e)}")
            return None
    
    def setup_gdrive_remote(self, folder_id: str):
        """
        Setup Google Drive as DVC remote
        
        Args:
            folder_id: Google Drive folder ID
        """
        gdrive_url = f"gdrive://{folder_id}"
        self.add_remote("gdrive", gdrive_url, set_default=True)
        
        # Configure gdrive settings
        try:
            subprocess.run([
                "dvc", "remote", "modify", "gdrive", "gdrive_acknowledge_abuse", "true"
            ], cwd=self.project_root)
            
            logger.info("Google Drive remote configured successfully")
            logger.info("Please run 'dvc remote modify gdrive gdrive_use_service_account true' if using service account")
            
        except Exception as e:
            logger.error(f"Error configuring Google Drive remote: {str(e)}")


def create_dvc_pipeline_config():
    """
    Create DVC pipeline configuration for SENSEX project
    """
    pipeline_config = {
        'stages': {
            'fetch_data': {
                'cmd': 'python src/data/fetch_data.py',
                'deps': [
                    'src/data/fetch_data.py',
                    'configs/config.py'
                ],
                'outs': [
                    'data/raw/'
                ]
            },
            'create_features': {
                'cmd': 'python src/data/create_feature_maps.py',
                'deps': [
                    'src/data/create_feature_maps.py',
                    'data/raw/'
                ],
                'outs': [
                    'data/processed/feature_maps.npy',
                    'data/processed/targets.npy',
                    'data/processed/dates.csv',
                    'data/processed/feature_names.csv',
                    'data/processed/stock_names.csv'
                ]
            },
            'train_model': {
                'cmd': 'python src/models/convlstm_model.py',
                'deps': [
                    'src/models/convlstm_model.py',
                    'data/processed/feature_maps.npy',
                    'data/processed/targets.npy'
                ],
                'outs': [
                    'models/convlstm_sensex_final.h5'
                ],
                'metrics': [
                    'metrics.json'
                ]
            }
        }
    }
    
    return pipeline_config


def setup_project_dvc():
    """
    Setup DVC for the SENSEX project
    """
    # Initialize DVC manager
    dvc_manager = DVCManager()
    
    # Initialize DVC
    dvc_manager.init_dvc()
    
    # Create pipeline configuration
    pipeline_config = create_dvc_pipeline_config()
    dvc_manager.create_pipeline(pipeline_config)
    
    # Add data directories to DVC tracking
    data_dirs = [
        'data/raw',
        'data/processed',
        'models'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            dvc_manager.add_data(data_dir)
    
    logger.info("DVC setup completed!")
    logger.info("To setup Google Drive remote:")
    logger.info("1. Create a folder in Google Drive")
    logger.info("2. Get the folder ID from the URL")
    logger.info("3. Run: python -c 'from src.utils.dvc_utils import DVCManager; dvc = DVCManager(); dvc.setup_gdrive_remote(\"YOUR_FOLDER_ID\")'")


def main():
    """
    Main function to setup DVC
    """
    setup_project_dvc()


if __name__ == "__main__":
    main()
