
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
