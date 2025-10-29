# Hugging Face Spaces Deployment Script
# Run this script to prepare your repository for Hugging Face Spaces deployment

import os
import shutil
import json
from pathlib import Path

def create_spaces_config():
    """Create Hugging Face Spaces configuration"""
    spaces_config = """title: SENSEX Next-Day Forecast
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-powered SENSEX market prediction using ConvLSTM neural networks
tags:
  - machine-learning
  - finance
  - stock-prediction
  - time-series
  - streamlit
  - tensorflow
  - mlops
"""
    
    with open("README.md", "w") as f:
        f.write(spaces_config)
    
    print("✅ Created README.md for Hugging Face Spaces")

def prepare_dockerfile():
    """Ensure Dockerfile is ready for Spaces"""
    # Copy the Streamlit Dockerfile as main Dockerfile
    if os.path.exists("Dockerfile.streamlit"):
        shutil.copy("Dockerfile.streamlit", "Dockerfile")
        print("✅ Copied Dockerfile.streamlit to Dockerfile")
    else:
        print("❌ Dockerfile.streamlit not found")

def create_app_file():
    """Create main app.py file for Spaces"""
    # Copy deployment app as main app
    if os.path.exists("streamlit_app/app_deployment.py"):
        shutil.copy("streamlit_app/app_deployment.py", "app.py")
        print("✅ Created app.py from deployment version")
    else:
        print("❌ app_deployment.py not found")

def copy_requirements():
    """Copy requirements file"""
    if os.path.exists("streamlit_app/requirements.txt"):
        shutil.copy("streamlit_app/requirements.txt", "requirements.txt")
        print("✅ Copied requirements.txt")
    else:
        print("❌ streamlit_app/requirements.txt not found")

def create_spaces_readme():
    """Create detailed README for the Space"""
    if os.path.exists("app_README.md"):
        # Append the app README content to the config
        with open("app_README.md", "r") as f:
            content = f.read()
        
        # Add spaces config at the top
        spaces_config = """---
title: SENSEX Next-Day Forecast
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

"""
        
        with open("README.md", "w") as f:
            f.write(spaces_config + content)
        
        print("✅ Created comprehensive README.md for Spaces")

def validate_deployment():
    """Validate deployment files"""
    required_files = [
        "README.md",
        "Dockerfile", 
        "app.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def create_deployment_instructions():
    """Create deployment instructions"""
    instructions = """
# 🚀 Hugging Face Spaces Deployment Instructions

## Steps to Deploy:

### 1. Create Hugging Face Account
- Go to https://huggingface.co and create a free account
- Verify your email address

### 2. Create New Space
- Click "New" → "Space"
- Choose a name: `your-username-sensex-predictor`
- Select SDK: **Docker**
- Set visibility: Public or Private
- Click "Create Space"

### 3. Connect to GitHub Repository
- In your Space settings, connect to your GitHub repo
- Enable automatic deployment on push

### 4. Upload Files
If not using GitHub integration, upload these files to your Space:
- `README.md` (Spaces configuration)
- `Dockerfile` (Container configuration)  
- `app.py` (Main application)
- `requirements.txt` (Python dependencies)
- `src/` folder (Source code)
- `streamlit_app/` folder (App assets)

### 5. Deploy
- Push to your connected GitHub repo, or
- Upload files directly to Spaces
- Spaces will automatically build and deploy

### 6. Access Your App
Your live app will be available at:
`https://your-username-sensex-predictor.hf.space`

## Environment Variables (Optional)
- `STREAMLIT_SERVER_PORT=7860`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

## Troubleshooting
- Check build logs in Spaces interface
- Ensure Dockerfile uses port 7860
- Verify all dependencies in requirements.txt
- Check that app.py runs locally first

## Demo Features
- Interactive SENSEX prediction
- Real-time market data (when available)
- Production-ready UI/UX
- System health monitoring

Your app will showcase the complete MLOps pipeline!
"""
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(instructions)
    
    print("✅ Created DEPLOYMENT_GUIDE.md")

def main():
    """Main deployment preparation function"""
    print("🚀 Preparing SENSEX MLOps project for Hugging Face Spaces deployment...")
    print()
    
    # Run preparation steps
    create_spaces_config()
    prepare_dockerfile() 
    create_app_file()
    copy_requirements()
    create_spaces_readme()
    create_deployment_instructions()
    
    print()
    print("📊 Validation...")
    is_valid = validate_deployment()
    
    print()
    if is_valid:
        print("🎉 Deployment preparation complete!")
        print()
        print("Next steps:")
        print("1. Create a Hugging Face Spaces account")
        print("2. Create a new Docker Space")
        print("3. Upload files or connect GitHub repo")
        print("4. Wait for automatic deployment")
        print("5. Access your live app!")
        print()
        print("📚 Read DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print("❌ Deployment preparation failed. Please fix missing files.")

if __name__ == "__main__":
    main()
