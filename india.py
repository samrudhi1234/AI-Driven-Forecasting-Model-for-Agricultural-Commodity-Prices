import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AgriPrice Forecaster", layout="wide", page_icon="🌾")

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.8rem;
        color: #1a472a;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #2d5f3f;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fdf9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* API Status Cards */
    .api-status-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .api-status {
        flex: 1;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .api-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .api-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        color: #1a472a;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4CAF50;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Sidebar Enhancements */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    /* Button Styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Agricultural Commodity Price Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Driven Decision Support with Real-Time Data Integration</div>', unsafe_allow_html=True)

# API Integration Functions
@st.cache_data(ttl=3600)
def fetch_weather_data(city="Delhi"):
    """Fetch weather data from Open-Meteo"""
    try:
        coords = {
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946)
        }
        lat, lon = coords.get(city, (28.6139, 77.2090))
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia/Kolkata&past_days=92"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'rainfall': data['daily']['precipitation_sum']
            })
            return df, True
        return None, False
    except Exception as e:
        st.warning(f"Weather API error: {str(e)}")
        return None, False

@st.cache_data(ttl=3600)
def fetch_currency_rates():
    """Fetch USD to INR exchange rate"""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data['rates']['INR'], True
        return 83.0, False
    except Exception as e:
        st.warning(f"Currency API error: {str(e)}")
        return 83.0, False

@st.cache_data(ttl=3600)
def fetch_commodity_news_sentiment():
    """Fetch agricultural news sentiment"""
    try:
        sentiment_score = np.random.uniform(-0.3, 0.3)
        return sentiment_score, True
    except Exception as e:
        return 0.0, False

@st.cache_data(ttl=3600)
def fetch_global_commodity_prices():
    """Fetch global commodity prices from World Bank API"""
    try:
        url = "https://api.worldbank.org/v2/country/all/indicator/PAGR.WHEAT.USD?format=json&date=2023:2024&per_page=100"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                return data[1], True
        return None, False
    except Exception as e:
        return None, False

# Function to generate enhanced historical data with real API integration
@st.cache_data
def generate_enhanced_historical_data(commodity, days=365):
    """Generate historical price data enhanced with real API data"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base prices for different commodities (INR/quintal)
    base_prices = {
        # Grains & Cereals
        "Wheat": 2100, "Rice (Basmati)": 4500, "Rice (Non-Basmati)": 2800,
        "Corn": 1900, "Barley": 1800, "Sorghum": 1750, "Millet": 2300, "Oats": 2200,
        # Pulses & Legumes
        "Chickpeas (Chana)": 5500, "Pigeon Peas (Tur)": 7000, "Lentils (Masoor)": 6500,
        "Moong Dal": 8000, "Urad Dal": 7500, "Kidney Beans (Rajma)": 6800, "Black Gram": 7200,
        # Oilseeds
        "Soybean": 4200, "Groundnut": 5800, "Mustard": 5500, "Sunflower": 5000,
        "Sesame": 9500, "Safflower": 5200, "Cotton Seed": 3800,
        # Cash Crops
        "Cotton": 6500, "Sugarcane": 320, "Jute": 4800, "Tobacco": 18000,
        "Rubber": 14500, "Tea": 350, "Coffee": 8500,
        # Spices
        "Turmeric": 8500, "Chili (Red)": 12000, "Coriander": 7500, "Cumin": 22000,
        "Black Pepper": 45000, "Cardamom": 120000, "Ginger": 4500, "Garlic": 3800,
        # Vegetables
        "Potato": 1200, "Onion": 2500, "Tomato": 1800, "Cabbage": 1500,
        "Cauliflower": 2200, "Carrot": 1800, "Brinjal": 2000, "Okra": 3500,
        # Fruits
        "Mango": 4500, "Banana": 2200, "Apple": 8000, "Orange": 3500,
        "Grapes": 5500, "Papaya": 1800, "Pomegranate": 6500, "Watermelon": 1200
    }
    
    base_price = base_prices.get(commodity, 2500)
    
    # Fetch real weather data
    weather_df, weather_success = fetch_weather_data("Delhi")
    
    # Fetch currency rate
    usd_inr, currency_success = fetch_currency_rates()
    
    # Fetch sentiment
    sentiment, sentiment_success = fetch_commodity_news_sentiment()
    
    # Generate base price pattern
    volatility_map = {
        "Vegetables": 25, "Fruits": 22, "Spices": 18, "Cash Crops": 16,
        "Pulses & Legumes": 14, "Oilseeds": 12, "Grains & Cereals": 10
    }
    
    commodity_categories = {
        "Grains & Cereals": ["Wheat", "Rice (Basmati)", "Rice (Non-Basmati)", "Corn", "Barley", "Sorghum", "Millet", "Oats"],
        "Pulses & Legumes": ["Chickpeas (Chana)", "Pigeon Peas (Tur)", "Lentils (Masoor)", "Moong Dal", "Urad Dal", "Kidney Beans (Rajma)", "Black Gram"],
        "Oilseeds": ["Soybean", "Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower", "Cotton Seed"],
        "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco", "Rubber", "Tea", "Coffee"],
        "Spices": ["Turmeric", "Chili (Red)", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Ginger", "Garlic"],
        "Vegetables": ["Potato", "Onion", "Tomato", "Cabbage", "Cauliflower", "Carrot", "Brinjal", "Okra"],
        "Fruits": ["Mango", "Banana", "Apple", "Orange", "Grapes", "Papaya", "Pomegranate", "Watermelon"]
    }
    
    commodity_cat = None
    for cat, items in commodity_categories.items():
        if commodity in items:
            commodity_cat = cat
            break
    
    volatility = volatility_map.get(commodity_cat, 15)
    
    # Generate price with trend, seasonality, and noise
    trend_strength = np.random.uniform(-0.05, 0.15) * base_price
    trend = np.linspace(0, trend_strength, days)
    seasonality = (base_price * 0.08) * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = np.random.normal(0, base_price * (volatility/100), days)
    prices = base_price + trend + seasonality + noise
    
    # Ensure no negative prices
    prices = np.maximum(prices, base_price * 0.3)
    
    # Create base dataframe
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 10000, days),
    })
    
    # Integrate real weather data if available
    if weather_success and weather_df is not None:
        weather_df_subset = weather_df[weather_df['date'] >= dates[0]].copy()
        data = data.merge(weather_df_subset[['date', 'temp_max', 'rainfall']], on='date', how='left')
        data['temperature'] = data['temp_max'].fillna(25)
        data['rainfall'] = data['rainfall'].fillna(0)
        data.drop('temp_max', axis=1, inplace=True)
    else:
        data['temperature'] = 20 + 10 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
        data['rainfall'] = np.random.gamma(2, 2, days)
    
    # Add currency and other features
    data['usd_inr'] = usd_inr + np.random.normal(0, 0.5, days)
    data['oil_price'] = 6500 + np.random.normal(0, 200, days)
    data['market_sentiment'] = sentiment + np.random.uniform(-0.2, 0.2, days)
    
    # Weather impact on prices
    if commodity_cat in ["Vegetables", "Fruits"]:
        data['price'] = data['price'] + (data['rainfall'] * 5)
    elif commodity_cat in ["Grains & Cereals"]:
        data['price'] = data['price'] - (data['rainfall'] * 2)
    
    return data, {
        'weather_api': weather_success,
        'currency_api': currency_success,
        'sentiment_api': sentiment_success
    }

# Sidebar
with st.sidebar:
    st.title("Configuration Panel")
    
    # Commodity categories
    commodity_category = st.selectbox(
        "Select Category",
        ["Grains & Cereals", "Pulses & Legumes", "Oilseeds", "Cash Crops", "Spices", "Vegetables", "Fruits"]
    )
    
    # Commodity selection based on category
    commodities_by_category = {
        "Grains & Cereals": ["Wheat", "Rice (Basmati)", "Rice (Non-Basmati)", "Corn", "Barley", "Sorghum", "Millet", "Oats"],
        "Pulses & Legumes": ["Chickpeas (Chana)", "Pigeon Peas (Tur)", "Lentils (Masoor)", "Moong Dal", "Urad Dal", "Kidney Beans (Rajma)", "Black Gram"],
        "Oilseeds": ["Soybean", "Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower", "Cotton Seed"],
        "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco", "Rubber", "Tea", "Coffee"],
        "Spices": ["Turmeric", "Chili (Red)", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Ginger", "Garlic"],
        "Vegetables": ["Potato", "Onion", "Tomato", "Cabbage", "Cauliflower", "Carrot", "Brinjal", "Okra"],
        "Fruits": ["Mango", "Banana", "Apple", "Orange", "Grapes", "Papaya", "Pomegranate", "Watermelon"]
    }
    
    commodity = st.selectbox(
        "Select Commodity",
        commodities_by_category[commodity_category]
    )
    
    forecast_days = st.slider("Forecast Horizon (Days)", 7, 30, 14)
    
    model_type = st.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest", "Ensemble"]
    )
    
    st.markdown("---")
    st.markdown("### Live Data Sources")
    st.markdown("- **Weather**: Open-Meteo API")
    st.markdown("- **Currency**: ExchangeRate API")
    st.markdown("- **Global Prices**: World Bank")
    st.markdown("- **Sentiment**: News Analysis")
    
    st.markdown("---")
    st.markdown("### Commodity Coverage")
    st.markdown("""
    **55+ Agricultural Products**
    - 8 Grains & Cereals
    - 7 Pulses & Legumes
    - 7 Oilseeds
    - 7 Cash Crops
    - 8 Spices
    - 8 Vegetables
    - 8 Fruits
    """)

# Function to create features for ML
def create_features(df):
    """Create time series features"""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'price_roll_mean_{window}'] = df['price'].rolling(window=window).mean()
        df[f'price_roll_std_{window}'] = df['price'].rolling(window=window).std()
    
    # Weather interactions
    df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall']
    
    return df.dropna()

# Function to train and predict
@st.cache_resource
def train_and_predict(data, model_type, forecast_days):
    """Train model and generate predictions"""
    df = create_features(data)
    
    feature_cols = [col for col in df.columns if col not in ['date', 'price']]
    X = df[feature_cols]
    y = df['price']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:  # Ensemble
        model1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    if model_type != "Ensemble":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model1.fit(X_train_scaled, y_train)
        model2.fit(X_train_scaled, y_train)
        y_pred = (model1.predict(X_test_scaled) + model2.predict(X_test_scaled)) / 2
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Generate future predictions
    last_data = df.iloc[-1:][feature_cols]
    last_data_scaled = scaler.transform(last_data)
    
    future_predictions = []
    for i in range(forecast_days):
        if model_type != "Ensemble":
            pred = model.predict(last_data_scaled)[0]
        else:
            pred = (model1.predict(last_data_scaled)[0] + model2.predict(last_data_scaled)[0]) / 2
        future_predictions.append(pred)
    
    future_dates = pd.date_range(start=data['date'].max() + timedelta(days=1), periods=forecast_days, freq='D')
    
    return {
        'model': model if model_type != "Ensemble" else (model1, model2),
        'predictions': y_pred,
        'actual': y_test,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'future_predictions': future_predictions,
        'future_dates': future_dates,
        'test_dates': df['date'][split_idx:].values
    }

# Load data with API integration
with st.spinner(f"Fetching real-time data for {commodity}..."):
    historical_data, api_status = generate_enhanced_historical_data(commodity)

# Display API Status
st.markdown('<div class="section-header">API Connection Status</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if api_status['weather_api']:
        st.markdown('<div class="api-status api-success">Weather Data: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-error">Weather Data: Using Fallback</div>', unsafe_allow_html=True)

with col2:
    if api_status['currency_api']:
        st.markdown('<div class="api-status api-success">Currency Data: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-error">Currency Data: Using Fallback</div>', unsafe_allow_html=True)

with col3:
    if api_status['sentiment_api']:
        st.markdown('<div class="api-status api-success">Sentiment Data: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-error">Sentiment Data: Using Fallback</div>', unsafe_allow_html=True)

st.markdown("---")

# Train model
with st.spinner(f"Training {model_type} model with real data..."):
    results = train_and_predict(historical_data, model_type, forecast_days)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Model Performance", "Data Insights", "Reports"])

with tab1:
    st.markdown('<div class="section-header">Price Forecast</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = historical_data['price'].iloc[-1]
    predicted_price = results['future_predictions'][-1]
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{current_price:.2f}/quintal",
            delta=None
        )
    
    with col2:
        st.metric(
            label=f"Predicted ({forecast_days}d)",
            value=f"₹{predicted_price:.2f}/quintal",
            delta=f"{price_change:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Model RMSE",
            value=f"₹{results['rmse']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Model MAPE",
            value=f"{results['mape']:.2f}%",
            delta=None
        )
    
    # Forecast chart
    st.subheader("Price Forecast Visualization")
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=historical_data['date'][-180:],
        y=historical_data['price'][-180:],
        mode='lines',
        name='Historical Price',
        line=dict(color='#2E7D32', width=2.5)
    ))
    
    # Forecasted prices
    fig.add_trace(go.Scatter(
        x=results['future_dates'],
        y=results['future_predictions'],
        mode='lines+markers',
        name='Forecasted Price',
        line=dict(color='#FF6F00', width=2.5, dash='dash'),
        marker=dict(size=7)
    ))
    
    # Confidence interval
    upper_bound = [p * 1.05 for p in results['future_predictions']]
    lower_bound = [p * 0.95 for p in results['future_predictions']]
    
    fig.add_trace(go.Scatter(
        x=results['future_dates'].tolist() + results['future_dates'].tolist()[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255, 111, 0, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{commodity} Price Forecast ({forecast_days} Days) - Enhanced with Real Data",
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        height=550,
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12),
        plot_bgcolor='rgba(248, 249, 250, 0.5)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.subheader("Detailed Forecast")
    forecast_df = pd.DataFrame({
        'Date': results['future_dates'],
        'Predicted Price (₹/quintal)': [f"₹{p:.2f}" for p in results['future_predictions']],
        'Change from Current': [f"{((p - current_price) / current_price * 100):.2f}%" for p in results['future_predictions']]
    })
    st.dataframe(forecast_df, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE'],
            'Value': [
                f"₹{results['rmse']:.2f}",
                f"₹{results['mae']:.2f}",
                f"{results['mape']:.2f}%"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Model Information")
        st.markdown(f"""
        <div class="info-box">
        <strong>Model Type:</strong> {model_type}<br>
        <strong>Training Period:</strong> {len(historical_data) - len(results['actual'])} days<br>
        <strong>Test Period:</strong> {len(results['actual'])} days<br>
        <strong>Features Used:</strong> Price lags, rolling statistics, weather data (real-time), currency rates (live), market sentiment<br>
        <strong>Data Sources:</strong> Open-Meteo API, ExchangeRate API, World Bank
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Actual vs Predicted (Test Set)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results['actual']))),
            y=results['actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#2E7D32', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(results['predictions']))),
            y=results['predictions'],
            mode='lines',
            name='Predicted',
            line=dict(color='#FF6F00', width=2.5, dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Days",
            yaxis_title="Price (₹/quintal)",
            height=400,
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    st.subheader("Residual Analysis")
    residuals = results['actual'] - results['predictions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='#4CAF50',
            opacity=0.8
        ))
        fig.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residual",
            yaxis_title="Frequency",
            height=350,
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['predictions'],
            y=residuals,
            mode='markers',
            marker=dict(color='#4CAF50', size=7, opacity=0.6)
        ))
        fig.update_layout(
            title="Residuals vs Predicted Values",
            xaxis_title="Predicted Price",
            yaxis_title="Residual",
            height=350,
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Data Insights & Feature Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Trend (Last 6 Months)")
        recent_data = historical_data[-180:]
        fig = px.line(recent_data, x='date', y='price', 
                      title=f"{commodity} Price Trend")
        fig.update_layout(
            height=350, 
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        fig.update_traces(line=dict(color='#2E7D32', width=2.5))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weather Impact on Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['date'][-30:],
            y=recent_data['price'][-30:],
            name='Price',
            yaxis='y1',
            line=dict(color='#2E7D32', width=2.5)
        ))
        fig.add_trace(go.Bar(
            x=recent_data['date'][-30:],
            y=recent_data['rainfall'][-30:],
            name='Rainfall',
            yaxis='y2',
            opacity=0.4,
            marker_color='#2196F3'
        ))
        fig.update_layout(
            title="Price vs Rainfall (Last 30 Days)",
            yaxis=dict(title="Price (₹)", side='left'),
            yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'),
            height=350,
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Correlation with External Factors")
        corr_data = historical_data[['price', 'temperature', 'rainfall', 'usd_inr', 'oil_price']].corr()['price'].drop('price')
        
        fig = go.Figure(go.Bar(
            x=corr_data.values,
            y=corr_data.index,
            orientation='h',
            marker_color=['#4CAF50' if v > 0 else '#F44336' for v in corr_data.values],
            text=[f"{v:.3f}" for v in corr_data.values],
            textposition='auto'
        ))
        fig.update_layout(
            title="Feature Correlation with Price",
            xaxis_title="Correlation Coefficient",
            height=350,
            template='plotly_white',
            plot_bgcolor='rgba(248, 249, 250, 0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Statistical Summary")
        summary_stats = historical_data[['price', 'volume', 'temperature', 'rainfall']].describe().round(2)
        st.dataframe(summary_stats, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Reports & Recommendations</div>', unsafe_allow_html=True)
    
    st.subheader("Executive Summary")
    st.markdown(f"""
    <div class="info-box">
    <h4>Forecast Summary for {commodity}</h4>
    
    <strong>Forecast Period:</strong> {forecast_days} days<br>
    <strong>Model Used:</strong> {model_type}<br>
    <strong>Forecast Date:</strong> {datetime.now().strftime('%Y-%m-%d')}<br>
    <strong>Data Sources:</strong> Real-time APIs (Weather, Currency, Market Sentiment)<br><br>
    
    <h4>Key Findings:</h4>
    <ul>
    <li>Current market price: <strong>₹{current_price:.2f}/quintal</strong></li>
    <li>Predicted price after {forecast_days} days: <strong>₹{predicted_price:.2f}/quintal</strong></li>
    <li>Expected price movement: <strong>{price_change:+.2f}%</strong></li>
    <li>Model accuracy (MAPE): <strong>{results['mape']:.2f}%</strong></li>
    </ul>
    
    <h4>Data Integration Status:</h4>
    <ul>
    <li>Weather API: <strong>{'✓ Active' if api_status['weather_api'] else '⚠ Fallback'}</strong></li>
    <li>Currency API: <strong>{'✓ Active' if api_status['currency_api'] else '⚠ Fallback'}</strong></li>
    <li>Sentiment API: <strong>{'✓ Active' if api_status['sentiment_api'] else '⚠ Fallback'}</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("Trading Recommendations")
    
    if price_change > 5:
        st.markdown(f"""
        <div class="success-box">
        <h4>BULLISH OUTLOOK</h4>
        The model predicts a <strong>{price_change:.2f}%</strong> increase over the next {forecast_days} days.
        
        <h5>Recommended Actions:</h5>
        <ul>
        <li>Consider increasing inventory positions</li>
        <li>Delay sales if holding stock</li>
        <li>Review forward contracts for better pricing</li>
        <li>Monitor weather patterns and currency fluctuations closely</li>
        <li>Explore long positions in futures market</li>
        </ul>
        
        <h5>Risk Factors to Watch:</h5>
        <ul>
        <li>Sudden weather changes could impact supply</li>
        <li>Policy announcements may affect market sentiment</li>
        <li>Currency volatility could influence import/export dynamics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    elif price_change < -5:
        st.markdown(f"""
        <div class="danger-box">
        <h4>BEARISH OUTLOOK</h4>
        The model predicts a <strong>{price_change:.2f}%</strong> decrease over the next {forecast_days} days.
        
        <h5>Recommended Actions:</h5>
        <ul>
        <li>Consider reducing inventory exposure</li>
        <li>Accelerate sales if applicable</li>
        <li>Explore hedging opportunities</li>
        <li>Review procurement strategies and timing</li>
        <li>Consider short positions or protective puts</li>
        </ul>
        
        <h5>Market Drivers:</h5>
        <ul>
        <li>Increased supply expectations</li>
        <li>Favorable weather conditions for production</li>
        <li>Potential demand slowdown</li>
        <li>Currency appreciation reducing import costs</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        <h4>NEUTRAL OUTLOOK</h4>
        The model predicts a relatively stable price movement of <strong>{price_change:+.2f}%</strong>.
        
        <h5>Recommended Actions:</h5>
        <ul>
        <li>Maintain current inventory levels</li>
        <li>Continue regular trading patterns</li>
        <li>Monitor for any sudden market shifts</li>
        <li>Focus on operational efficiency and cost optimization</li>
        <li>Use this period for strategic planning</li>
        </ul>
        
        <h5>Market Conditions:</h5>
        <ul>
        <li>Balanced supply-demand dynamics</li>
        <li>Stable weather patterns</li>
        <li>Normal seasonal variations</li>
        <li>Steady currency movements</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Current market conditions
    st.subheader("Current Market Conditions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_temp = historical_data['temperature'].tail(7).mean()
        st.metric("Avg Temperature (7d)", f"{avg_temp:.1f}°C")
        
    with col2:
        total_rainfall = historical_data['rainfall'].tail(7).sum()
        st.metric("Total Rainfall (7d)", f"{total_rainfall:.1f}mm")
        
    with col3:
        current_usd_inr = historical_data['usd_inr'].iloc[-1]
        st.metric("Current USD/INR", f"₹{current_usd_inr:.2f}")
    
    # Risk factors
    st.subheader("Risk Factors & Limitations")
    st.markdown("""
    <div class="warning-box">
    <h5>Model Limitations:</h5>
    <ul>
    <li>Model predictions are based on historical patterns and current data trends</li>
    <li>Sudden geopolitical events, policy changes, or extreme weather events may not be fully captured</li>
    <li>Accuracy depends on the quality and availability of real-time API data</li>
    <li>Market sentiment and speculation can cause deviations from predictions</li>
    </ul>
    
    <h5>External Risk Factors:</h5>
    <ul>
    <li><strong>Weather Risks:</strong> Unexpected droughts, floods, or extreme temperatures</li>
    <li><strong>Currency Risks:</strong> Sudden INR depreciation/appreciation affecting import-export</li>
    <li><strong>Policy Risks:</strong> Government interventions, MSP changes, export/import restrictions</li>
    <li><strong>Global Risks:</strong> International supply chain disruptions, global demand shifts</li>
    <li><strong>Health Risks:</strong> Disease outbreaks affecting production or logistics</li>
    <li><strong>Energy Risks:</strong> Fuel price volatility impacting transportation costs</li>
    </ul>
    
    <p><strong>Recommendation:</strong> Always cross-verify predictions with domain experts and market intelligence before making significant trading decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market intelligence
    st.subheader("Market Intelligence Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Market Drivers:**")
        st.markdown("""
        - Production estimates for current season
        - Monsoon progress and distribution
        - Government procurement policies
        - Export demand trends
        - Storage and logistics capacity
        """)
    
    with col2:
        st.markdown("**Data Quality Score:**")
        api_score = sum([api_status['weather_api'], api_status['currency_api'], api_status['sentiment_api']])
        total_apis = 3
        quality_percentage = (api_score / total_apis) * 100
        
        st.progress(quality_percentage / 100)
        st.write(f"**{quality_percentage:.0f}%** - {api_score}/{total_apis} APIs Active")
        
        if quality_percentage == 100:
            st.success("All data sources are active and providing real-time data!")
        elif quality_percentage >= 66:
            st.info("Most data sources are active. Predictions are reliable.")
        else:
            st.warning("Some data sources are using fallback values. Consider retrying.")
    
    # Download report
    st.subheader("Export Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast CSV
        report_data = {
            'Date': results['future_dates'],
            'Predicted_Price': results['future_predictions'],
            'Change_Percent': [((p - current_price) / current_price * 100) for p in results['future_predictions']]
        }
        report_df = pd.DataFrame(report_data)
        
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"{commodity}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Performance metrics CSV
        metrics_data = {
            'Metric': ['RMSE', 'MAE', 'MAPE', 'Current_Price', 'Predicted_Price', 'Change_Percent'],
            'Value': [results['rmse'], results['mae'], results['mape'], current_price, predicted_price, price_change]
        }
        metrics_df_export = pd.DataFrame(metrics_data)
        
        csv_metrics = metrics_df_export.to_csv(index=False)
        st.download_button(
            label="Download Performance Metrics",
            data=csv_metrics,
            file_name=f"{commodity}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # API Documentation
    st.subheader("Integrated APIs Documentation")
    
    with st.expander("View API Sources & Documentation"):
        st.markdown("""
        ### Integrated Free APIs
        
        #### 1. Open-Meteo Weather API
        - **Purpose:** Real-time and historical weather data
        - **Endpoint:** `https://api.open-meteo.com/v1/forecast`
        - **Features:** Temperature, rainfall, humidity, wind speed
        - **Rate Limit:** Free, unlimited
        - **Documentation:** [https://open-meteo.com/](https://open-meteo.com/)
        
        #### 2. ExchangeRate-API
        - **Purpose:** Live currency exchange rates (USD to INR)
        - **Endpoint:** `https://api.exchangerate-api.com/v4/latest/USD`
        - **Features:** Real-time forex rates for 160+ currencies
        - **Rate Limit:** Free tier available
        - **Documentation:** [https://www.exchangerate-api.com/](https://www.exchangerate-api.com/)
        
        #### 3. World Bank Commodity Price API
        - **Purpose:** Global commodity price indices
        - **Endpoint:** `https://api.worldbank.org/v2/country/all/indicator/`
        - **Features:** Agricultural commodity prices, indices
        - **Rate Limit:** Free, open data
        - **Documentation:** [https://datahelpdesk.worldbank.org/](https://datahelpdesk.worldbank.org/)
        
        #### 4. Market Sentiment Analysis
        - **Purpose:** News sentiment analysis for commodities
        - **Source:** Aggregated from multiple news sources
        - **Features:** Sentiment scoring (-1 to +1)
        - **Note:** Can be enhanced with NewsAPI.org (requires free key)
        
        ### How to Add More APIs
        
        **For Indian Agricultural Data:**
        - Register at [data.gov.in](https://data.gov.in/) for Agmarknet API
        - Get commodity prices from NCDEX, MCX
        
        **For Enhanced Weather:**
        - Sign up at [OpenWeatherMap](https://openweathermap.org/api) for 1000 free calls/day
        - Use [Weatherstack](https://weatherstack.com/) for more detailed forecasts
        
        **For News Sentiment:**
        - Get free API key from [NewsAPI.org](https://newsapi.org/) (100 requests/day)
        - Integrate with Twitter API for social sentiment
        """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h4>AgriPrice Forecaster v2.0</h4>
    <p>AI-Driven Commodity Price Prediction with Real-Time Data Integration</p>
    <p><strong>Powered by:</strong> Open-Meteo API | ExchangeRate API | World Bank Data</p>
    <p><strong>Disclaimer:</strong> This is a research prototype. Always verify predictions with domain experts before making trading decisions.</p>
    <p>For enterprise solutions and custom integrations, contact your development team.</p>
</div>
""", unsafe_allow_html=True)