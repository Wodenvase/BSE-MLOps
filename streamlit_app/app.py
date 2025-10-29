"""
SENSEX Next-Day Forecast - Interactive ML Application
Production-ready Streamlit app for SENSEX movement prediction using ConvLSTM
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import logging
import time
import traceback

# Add project root to path
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../src/serving')

# Import serving modules
try:
    from src.serving.model_registry import ModelRegistry
    from src.serving.model_server import ModelServer
    from src.serving.realtime_data import RealTimeDataFetcher
except ImportError as e:
    st.error(f"Could not import serving modules: {str(e)}")
    st.info("Make sure you're running from the correct directory and all dependencies are installed")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SENSEX Next-Day Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-up {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
        font-size: 2rem;
    }
    .prediction-down {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        font-weight: bold;
        font-size: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_server' not in st.session_state:
    st.session_state.model_server = None
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize core components
@st.cache_resource
def initialize_model_server():
    """Initialize and cache the model server"""
    try:
        registry = ModelRegistry()
        server = ModelServer(registry)
        
        if server.load_production_model():
            return server
        else:
            st.error("Failed to load production model")
            return None
    except Exception as e:
        st.error(f"Error initializing model server: {str(e)}")
        return None

@st.cache_resource
def initialize_data_fetcher():
    """Initialize and cache the data fetcher"""
    try:
        return RealTimeDataFetcher()
    except Exception as e:
        st.error(f"Error initializing data fetcher: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data():
    """Get latest market data"""
    try:
        if st.session_state.data_fetcher is None:
            st.session_state.data_fetcher = initialize_data_fetcher()
        
        if st.session_state.data_fetcher:
            return st.session_state.data_fetcher.get_market_summary()
        return {}
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return {}

def make_prediction():
    """Make a new prediction using the model server"""
    try:
        if st.session_state.model_server is None:
            st.session_state.model_server = initialize_model_server()
        
        if st.session_state.data_fetcher is None:
            st.session_state.data_fetcher = initialize_data_fetcher()
        
        if not st.session_state.model_server or not st.session_state.data_fetcher:
            st.error("Model server or data fetcher not available")
            return None
        
        # Fetch latest data
        with st.spinner("Fetching latest market data..."):
            raw_data = st.session_state.data_fetcher.fetch_latest_data(period="60d")
        
        if not raw_data:
            st.error("No market data available")
            return None
        
        # Validate data
        quality_report = st.session_state.data_fetcher.validate_data_quality(raw_data)
        
        if not quality_report.get('readiness', False):
            st.warning("Data quality issues detected:")
            for issue in quality_report.get('issues', []):
                st.warning(f"• {issue}")
            
            if quality_report.get('status') == 'invalid':
                st.error("Cannot make prediction due to data quality issues")
                return None
        
        # Make prediction
        with st.spinner("Running ConvLSTM model prediction..."):
            prediction = st.session_state.model_server.predict_from_raw_data(raw_data)
        
        if prediction:
            # Save to session state
            st.session_state.last_prediction = prediction
            st.session_state.prediction_history.append(prediction)
            
            # Keep only last 10 predictions
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[-10:]
            
            # Log prediction
            st.session_state.model_server.save_prediction_log(prediction)
        
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return None

def create_prediction_gauge(probability, prediction):
    """Create gauge chart for prediction confidence"""
    direction = "UP" if prediction == 1 else "DOWN"
    color = "#00C853" if prediction == 1 else "#D32F2F"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence - {direction}", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "white"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'suffix': "%", 'font': {'size': 20}}
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 SENSEX Next-Day Forecast</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Model Status")
        
        # Initialize components if needed
        if st.session_state.model_server is None:
            st.session_state.model_server = initialize_model_server()
        
        if st.session_state.data_fetcher is None:
            st.session_state.data_fetcher = initialize_data_fetcher()
        
        # Model status
        if st.session_state.model_server:
            model_info = st.session_state.model_server.get_model_info()
            st.success("✅ Model Loaded")
            if model_info.get('version'):
                st.info(f"Version: {model_info['version']}")
                if 'metrics' in model_info and 'test_accuracy' in model_info['metrics']:
                    st.info(f"Accuracy: {model_info['metrics']['test_accuracy']:.3f}")
        else:
            st.error("❌ Model Not Available")
        
        # Data status
        if st.session_state.data_fetcher:
            st.success("✅ Data Pipeline Ready")
        else:
            st.error("❌ Data Pipeline Error")
        
        st.markdown("---")
        
        # System health
        st.markdown("### 🏥 System Health")
        if st.button("🔍 Check Health"):
            if st.session_state.model_server:
                health = st.session_state.model_server.health_check()
                if health['status'] == 'healthy':
                    st.success("System Healthy")
                else:
                    st.error("System Issues Detected")
                    for check, status in health['checks'].items():
                        if not status:
                            st.warning(f"❌ {check}")
        
        # Clear cache
        if st.button("�️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            if st.session_state.data_fetcher:
                st.session_state.data_fetcher.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    # Main prediction interface
    st.markdown("## 🎯 Make Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **How it works:**
        1. Fetches latest 60 days of SENSEX component data
        2. Processes 40+ technical indicators 
        3. Uses ConvLSTM model to predict next-day movement
        4. Provides probability and confidence scores
        """)
    
    with col2:
        # Main prediction button
        predict_button = st.button(
            "🔮 Run Prediction",
            type="primary",
            help="Click to generate next-day SENSEX forecast",
            use_container_width=True
        )
    
    # Handle prediction
    if predict_button:
        prediction = make_prediction()
        
        if prediction:
            st.session_state.last_prediction = prediction
    
    # Display latest prediction
    if st.session_state.last_prediction:
        st.markdown("## 📊 Latest Forecast")
        
        prediction = st.session_state.last_prediction
        
        # Main prediction display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction = prediction['direction']
            probability = prediction['probability']
            
            if direction == 'UP':
                st.markdown(f"""
                <div class="prediction-card prediction-up">
                    <h2>📈 {direction}</h2>
                    <p>Next-day movement</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card prediction-down">
                    <h2>📉 {direction}</h2>
                    <p>Next-day movement</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Probability",
                f"{probability:.3f}",
                delta=f"{(probability - 0.5):+.3f}"
            )
            
            st.metric(
                "Confidence", 
                prediction['confidence'],
                delta=f"{prediction['confidence_score']:.2f}"
            )
        
        with col3:
            # Gauge chart
            gauge_fig = create_prediction_gauge(probability, prediction['prediction'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Prediction details
        st.markdown("### 📋 Prediction Details")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.info(f"""
            **Timestamp**: {prediction['timestamp'][:19]}
            **Model Version**: {prediction.get('model_version', 'N/A')}
            **Prediction Type**: Binary Classification
            """)
        
        with details_col2:
            # Risk assessment
            conf_score = prediction['confidence_score']
            if conf_score > 0.8:
                risk_level = "Very Low"
                risk_color = "green"
            elif conf_score > 0.6:
                risk_level = "Low"
                risk_color = "green"
            elif conf_score > 0.4:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "High"
                risk_color = "red"
            
            st.markdown(f"""
            **Risk Assessment**: <span style="color: {risk_color}; font-weight: bold">{risk_level}</span>
            
            **Recommendation**: {'Consider the prediction' if conf_score > 0.4 else 'Use with caution'}
            """, unsafe_allow_html=True)
    
    # Market overview
    if st.session_state.data_fetcher:
        st.markdown("## 📈 Market Overview")
        
        market_data = get_market_data()
        
        if market_data and 'sensex' in market_data:
            sensex_info = market_data['sensex']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "SENSEX",
                    f"{sensex_info.get('current_price', 0):,.0f}",
                    f"{sensex_info.get('change', 0):+.0f} ({sensex_info.get('change_percent', 0):+.1f}%)"
                )
            
            with col2:
                breadth = market_data.get('market_breadth', {})
                st.metric(
                    "Advancing",
                    breadth.get('advancing', 0),
                    f"vs {breadth.get('declining', 0)} declining"
                )
            
            with col3:
                if 'top_gainers' in market_data and market_data['top_gainers']:
                    top_gainer = market_data['top_gainers'][0]
                    st.metric(
                        "Top Gainer",
                        top_gainer['symbol'],
                        f"{top_gainer['change_percent']:+.1f}%"
                    )
            
            with col4:
                if 'top_losers' in market_data and market_data['top_losers']:
                    top_loser = market_data['top_losers'][-1]
                    st.metric(
                        "Top Loser", 
                        top_loser['symbol'],
                        f"{top_loser['change_percent']:+.1f}%"
                    )
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("## � Recent Predictions")
        
        history_df = pd.DataFrame([
            {
                'Timestamp': pred['timestamp'][:19],
                'Direction': pred['direction'],
                'Probability': f"{pred['probability']:.3f}",
                'Confidence': pred['confidence']
            }
            for pred in st.session_state.prediction_history[-5:]  # Last 5 predictions
        ])
        
        st.dataframe(history_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🚀 <strong>SENSEX Next-Day Forecast</strong> | Powered by ConvLSTM & MLflow</p>
        <p>⚠️ <em>This is for educational and research purposes only. Not financial advice.</em></p>
        <p>🔬 Built with TensorFlow, MLflow, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
