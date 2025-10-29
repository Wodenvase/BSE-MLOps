# Streamlit App Structure

## Main Application Files

### 🎯 **Primary Streamlit App**
- **File:** `app.py` (Root directory)
- **Size:** 19,703 bytes
- **Purpose:** Main production deployment version
- **Features:**
  - Optimized for Hugging Face Spaces deployment
  - Comprehensive fallback mechanisms
  - Demo mode for cloud deployment
  - Full MLOps pipeline demonstration

### 📦 **Secondary App** 
- **File:** `streamlit_app/app.py` (Subdirectory)
- **Size:** 16,045 bytes
- **Purpose:** Alternative/development version
- **Status:** Backup version in subdirectory

## 🚀 **For Deployment Use:**

### **Streamlit Cloud Deployment:**
```yaml
Repository: Wodenvase/BSE-MLOps
Branch: main
Main file path: app.py  # Use the root app.py
```

### **Local Development:**
```bash
# Run main app
streamlit run app.py

# Or run alternative app
streamlit run streamlit_app/app.py
```

### **Docker Deployment:**
```bash
# Uses app.py by default
docker build -t bse-mlops .
docker run -p 8501:8501 bse-mlops
```

## 📁 **File Organization:**

```
BSE-MLOps/
├── app.py                    # 🎯 MAIN STREAMLIT APP
├── streamlit_app/
│   ├── app.py               # Alternative version
│   ├── app_deployment.py    # Deployment utilities
│   ├── requirements.txt     # App-specific requirements
│   └── run_app.sh          # Run script
├── requirements.txt         # 🎯 MAIN REQUIREMENTS
├── runtime.txt             # Python version for cloud
├── packages.txt            # System dependencies
└── deploy_streamlit.sh     # Deployment script
```

## ✅ **Current Status:**
- Main app: `app.py` ✅ Ready for deployment
- Requirements: ✅ Updated and committed
- Runtime config: ✅ Python 3.9 specified
- Repository: ✅ All changes pushed to GitHub
- Compatibility: ✅ Fixed for Streamlit Cloud

## 🎮 **Usage:**
The main `app.py` file is your production-ready Streamlit application that includes:
- Complete BSE MLOps pipeline demonstration
- Real-time SENSEX prediction capabilities
- Interactive market data visualization
- Model performance monitoring
- Comprehensive demo mode for cloud deployment
