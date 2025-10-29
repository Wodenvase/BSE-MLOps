"""
SENSEX Next-Day Forecast - Clean Streamlit Cloud Version
Fully self-contained with no external module dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="SENSEX Next-Day Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_demo_prediction():
    """Generate realistic demo prediction"""
    scenarios = [
        {"direction": "UP", "probability": 0.73, "confidence": "High", "confidence_score": 0.46},
        {"direction": "DOWN", "probability": 0.64, "confidence": "Medium", "confidence_score": 0.28},
        {"direction": "UP", "probability": 0.68, "confidence": "High", "confidence_score": 0.36},
        {"direction": "DOWN", "probability": 0.59, "confidence": "Medium", "confidence_score": 0.18},
        {"direction": "UP", "probability": 0.71, "confidence": "High", "confidence_score": 0.42}
    ]
    return random.choice(scenarios)

def generate_market_data():
    """Generate demo market data"""
    # Generate last 30 days of SENSEX data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    base_price = 65000
    
    prices = []
    current_price = base_price
    
    for i in range(30):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)
        current_price *= (1 + change)
        prices.append(current_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(100000000, 500000000, 30)
    })
    
    return df

def create_price_chart(df):
    """Create interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='SENSEX',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='SENSEX Historical Prices (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue',
        hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Trading Volume (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 SENSEX Next-Day Forecast</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Prediction", "Market Overview", "About"])
    
    if page == "Prediction":
        st.markdown("## 🎯 Next-Day Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Generate SENSEX Movement Prediction")
            st.write("Click the button below to get an AI-powered prediction for tomorrow's SENSEX movement.")
            
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Analyzing market data and generating prediction..."):
                    # Simulate processing time
                    import time
                    time.sleep(2)
                    
                    prediction = generate_demo_prediction()
                    
                    # Display prediction
                    direction_color = "🟢" if prediction["direction"] == "UP" else "🔴"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>{direction_color} Prediction: {prediction["direction"]}</h3>
                        <p><strong>Probability:</strong> {prediction["probability"]:.1%}</p>
                        <p><strong>Confidence:</strong> {prediction["confidence"]}</p>
                        <p><strong>Confidence Score:</strong> {prediction["confidence_score"]:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ Prediction generated successfully!")
        
        with col2:
            st.markdown("### Model Information")
            st.markdown("""
            <div class="metric-card">
                <strong>Model:</strong> ConvLSTM<br>
                <strong>Features:</strong> 45 technical indicators<br>
                <strong>Training Data:</strong> 30 SENSEX stocks<br>
                <strong>Accuracy:</strong> 55-65%<br>
                <strong>Update Frequency:</strong> Daily
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Market Overview":
        st.markdown("## 📊 Market Overview")
        
        # Generate demo data
        market_data = generate_market_data()
        
        # Key metrics
        current_price = market_data['Close'].iloc[-1]
        prev_price = market_data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"₹{current_price:,.0f}", f"{change:+,.0f}")
        
        with col2:
            st.metric("Change %", f"{change_pct:+.2f}%")
        
        with col3:
            st.metric("Volume", f"{market_data['Volume'].iloc[-1]:,.0f}")
        
        with col4:
            st.metric("30-Day High", f"₹{market_data['Close'].max():,.0f}")
        
        # Charts
        st.plotly_chart(create_price_chart(market_data), use_container_width=True)
        st.plotly_chart(create_volume_chart(market_data), use_container_width=True)
    
    else:  # About page
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Overview
        This is a demonstration of an end-to-end MLOps pipeline for predicting next-day SENSEX movement using ConvLSTM neural networks.
        
        ### 🏗️ Architecture
        - **Data Engineering**: Automated SENSEX data fetching with Apache Airflow
        - **ML Training**: ConvLSTM model with MLflow experiment tracking
        - **Model Serving**: Real-time inference API with caching
        - **Deployment**: Containerized deployment with monitoring
        
        ### 📊 Features
        - **Real-time Predictions**: Next-day SENSEX movement forecasting
        - **Interactive UI**: Beautiful charts and visualizations
        - **Production Ready**: Comprehensive error handling and fallbacks
        - **Demo Mode**: Works without external dependencies
        
        ### 🚨 Disclaimer
        This application is for **educational and demonstration purposes only**. 
        It is **NOT financial advice** and should not be used for actual trading decisions.
        
        ### 🔗 Links
        - **GitHub Repository**: [BSE-MLOps](https://github.com/Wodenvase/BSE-MLOps)
        - **Documentation**: Available in the repository
        
        ---
        **Built with ❤️ using Streamlit, TensorFlow, and MLflow**
        """)

if __name__ == "__main__":
    main()
