"""
DVC Configuration and Google Drive Setup for SENSEX Data Pipeline
This file contains DVC configuration and helper scripts for data versioning
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVCManager:
    """Manages DVC operations for the SENSEX data pipeline"""
    
    def __init__(self, project_root: str = "/opt/airflow"):
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        
    def initialize_dvc(self) -> bool:
        """Initialize DVC in the project if not already initialized"""
        try:
            if not self.dvc_dir.exists():
                logger.info("Initializing DVC...")
                result = subprocess.run(
                    ["dvc", "init"], 
                    cwd=self.project_root, 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("✅ DVC initialized successfully")
                    return True
                else:
                    logger.error(f"❌ DVC initialization failed: {result.stderr}")
                    return False
            else:
                logger.info("✅ DVC already initialized")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error initializing DVC: {str(e)}")
            return False
    
    def setup_google_drive_remote(self, remote_name: str = "gdrive", 
                                 folder_id: Optional[str] = None) -> bool:
        """Setup Google Drive as DVC remote storage"""
        try:
            # Check if remote already exists
            result = subprocess.run(
                ["dvc", "remote", "list"], 
                cwd=self.project_root,
                capture_output=True, 
                text=True
            )
            
            if remote_name in result.stdout:
                logger.info(f"✅ Remote '{remote_name}' already exists")
                return True
            
            # Add Google Drive remote
            if folder_id:
                gdrive_url = f"gdrive://{folder_id}"
            else:
                # Use root folder if no specific folder ID provided
                gdrive_url = "gdrive://root"
            
            logger.info(f"Adding Google Drive remote: {gdrive_url}")
            result = subprocess.run([
                "dvc", "remote", "add", "-d", remote_name, gdrive_url
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Google Drive remote '{remote_name}' added successfully")
                
                # Set remote as default
                subprocess.run([
                    "dvc", "remote", "default", remote_name
                ], cwd=self.project_root)
                
                return True
            else:
                logger.error(f"❌ Failed to add Google Drive remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting up Google Drive remote: {str(e)}")
            return False
    
    def add_data_to_dvc(self, data_paths: List[str]) -> bool:
        """Add data files/directories to DVC tracking"""
        try:
            success = True
            
            for data_path in data_paths:
                full_path = self.project_root / data_path
                
                if not full_path.exists():
                    logger.warning(f"⚠️ Path does not exist: {data_path}")
                    continue
                
                logger.info(f"Adding to DVC: {data_path}")
                result = subprocess.run([
                    "dvc", "add", str(full_path)
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ Added to DVC: {data_path}")
                else:
                    logger.error(f"❌ Failed to add to DVC: {data_path} - {result.stderr}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error adding data to DVC: {str(e)}")
            return False
    
    def push_data_to_remote(self) -> bool:
        """Push DVC tracked data to remote storage"""
        try:
            logger.info("Pushing data to DVC remote...")
            result = subprocess.run([
                "dvc", "push"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Data pushed to remote successfully")
                return True
            else:
                logger.error(f"❌ Failed to push data to remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pushing data to remote: {str(e)}")
            return False
    
    def pull_data_from_remote(self) -> bool:
        """Pull DVC tracked data from remote storage"""
        try:
            logger.info("Pulling data from DVC remote...")
            result = subprocess.run([
                "dvc", "pull"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Data pulled from remote successfully")
                return True
            else:
                logger.error(f"❌ Failed to pull data from remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pulling data from remote: {str(e)}")
            return False
    
    def get_data_status(self) -> Dict:
        """Get DVC data status"""
        try:
            result = subprocess.run([
                "dvc", "status"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            return {
                "status_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting DVC status: {str(e)}")
            return {"error": str(e)}
    
    def create_pipeline_version_tag(self, version: str, message: str) -> bool:
        """Create a version tag for the data pipeline"""
        try:
            # First commit any changes
            subprocess.run([
                "git", "add", ".dvc/", "*.dvc", ".gitignore"
            ], cwd=self.project_root)
            
            subprocess.run([
                "git", "commit", "-m", f"Data pipeline version {version}: {message}"
            ], cwd=self.project_root)
            
            # Create tag
            result = subprocess.run([
                "git", "tag", "-a", f"data-v{version}", "-m", message
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Created version tag: data-v{version}")
                return True
            else:
                logger.error(f"❌ Failed to create version tag: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating version tag: {str(e)}")
            return False


def setup_dvc_pipeline():
    """Complete DVC setup for the SENSEX data pipeline"""
    dvc_manager = DVCManager()
    
    # Step 1: Initialize DVC
    if not dvc_manager.initialize_dvc():
        return False
    
    # Step 2: Setup Google Drive remote
    # You can replace this with your specific Google Drive folder ID
    if not dvc_manager.setup_google_drive_remote():
        return False
    
    # Step 3: Add initial data directories to DVC
    data_paths = [
        "data/raw",
        "data/processed", 
        "data/models"  # For future ML models
    ]
    
    # Create directories if they don't exist
    for path in data_paths:
        full_path = Path("/opt/airflow") / path
        full_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ DVC setup completed successfully")
    return True


if __name__ == "__main__":
    setup_dvc_pipeline()
