# Streamlit Cloud Deployment Troubleshooting Guide

## 🚨 Common Issues & Solutions

### Issue 1: "Branch does not exist" or "File does not exist"

#### Solution A: Wait and Refresh
1. **Wait 2-3 minutes** after pushing to GitHub
2. **Refresh** the Streamlit Cloud deployment page
3. **Clear browser cache** and try again

#### Solution B: Try Different Repository Format
Instead of: `https://github.com/Wodenvase/BSE-MLOps`
Try: `Wodenvase/BSE-MLOps`

#### Solution C: Use Repository Dropdown
1. Click the **dropdown arrow** next to the repository field
2. **Search for "Wodenvase"**
3. **Select "BSE-MLOps"** from the dropdown

### Issue 2: Deployment Fails Due to Requirements

#### Solution: Use Minimal Requirements
1. **Rename** current requirements.txt to requirements_full.txt
2. **Rename** requirements_minimal.txt to requirements.txt
3. **Deploy with minimal dependencies**

```bash
# In your local terminal:
cd /path/to/BSE-MLOps
mv requirements.txt requirements_full.txt
mv requirements_minimal.txt requirements.txt
git add .
git commit -m "Use minimal requirements for Streamlit Cloud"
git push
```

### Issue 3: Test with Simple App First

#### Solution: Deploy Test App
1. **Use test_app.py** instead of app.py
2. **Deploy settings:**
   - Repository: `Wodenvase/BSE-MLOps`
   - Branch: `main`
   - Main file path: `test_app.py`

### Issue 4: Memory or Timeout Issues

#### Solution: App Optimization
The main app.py includes fallback mechanisms, but if it's still too heavy:
1. **Use the test app first** to verify deployment works
2. **Gradually add features** back

## 🔧 Step-by-Step Deployment Process

### Step 1: Verify Repository Access
1. Go to https://github.com/Wodenvase/BSE-MLOps
2. Verify you can see all files including app.py
3. Check that the latest commits are visible

### Step 2: Clean Deployment Attempt
1. Go to https://share.streamlit.io/
2. **Sign out and sign back in**
3. Try deployment with these **exact settings:**

```
Repository: Wodenvase/BSE-MLOps
Branch: main
Main file path: app.py
```

### Step 3: Alternative - Test App First
If main app fails, try:
```
Repository: Wodenvase/BSE-MLOps
Branch: main
Main file path: test_app.py
```

### Step 4: Check Streamlit Cloud Status
1. Visit https://status.streamlit.io/
2. Check if there are any ongoing issues

## 🆘 Emergency Deployment Options

### Option 1: Use Streamlit Community Cloud
- Go to https://streamlit.io/cloud
- Same process but different platform

### Option 2: Deploy to Other Platforms

#### Render.com
1. Connect GitHub repository
2. Environment: Python 3
3. Build command: `pip install -r requirements_minimal.txt`
4. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

#### Railway.app
1. Connect GitHub repository  
2. Add environment variables if needed
3. Railway auto-detects Streamlit apps

## 🔍 Debugging Information

### Check These Files Exist:
- ✅ app.py (main application)
- ✅ test_app.py (simple test)
- ✅ requirements.txt (dependencies)
- ✅ requirements_minimal.txt (minimal deps)
- ✅ runtime.txt (Python version)
- ✅ packages.txt (system deps)

### Repository URL Formats to Try:
1. `https://github.com/Wodenvase/BSE-MLOps`
2. `https://github.com/Wodenvase/BSE-MLOps.git`
3. `Wodenvase/BSE-MLOps`
4. Use dropdown selection

## 📞 What to Tell Me

If still having issues, please share:
1. **Exact error message** from Streamlit Cloud
2. **Screenshot** of the deployment page
3. **Which step** is failing (repository detection, file detection, or actual deployment)
4. **Any console errors** in browser developer tools

This will help me provide more specific assistance!
