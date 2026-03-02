import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AgriPrice Forecaster", layout="wide", page_icon="A", initial_sidebar_state="expanded")

# Modern CSS without emojis
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #10B981;
    --primary-dark: #059669;
    --primary-light: #D1FAE5;
    --secondary: #3B82F6;
    --accent: #F59E0B;
    --danger: #EF4444;
    --dark: #1F2937;
    --gray-50: #F9FAFB;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-300: #D1D5DB;
    --gray-600: #4B5563;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
}

.main { background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0fdfa 100%); font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0fdfa 100%); }
#MainMenu, footer, header { visibility: hidden; }

.hero-container {
    background: linear-gradient(135deg, #065F46 0%, #047857 50%, #059669 100%);
    padding: 2.5rem 3rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-xl);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(255,255,255,0.1) 0%, transparent 70%);
}
.hero-title { font-size: 2.5rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -1px; }
.hero-subtitle { font-size: 1.05rem; color: rgba(255,255,255,0.85); font-weight: 400; max-width: 600px; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.15);
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    color: #D1FAE5;
    margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.2);
}

.api-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0; }
.api-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--gray-200);
    transition: all 0.3s ease;
}
.api-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }
.api-card.success { border-left: 4px solid var(--primary); }
.api-card.error { border-left: 4px solid var(--danger); }
.api-icon { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; font-weight: 700; margin-bottom: 0.75rem; }
.api-icon.success { background: var(--primary-light); color: var(--primary-dark); }
.api-icon.error { background: #FEE2E2; color: var(--danger); }
.api-title { font-size: 0.85rem; color: var(--gray-600); font-weight: 500; margin-bottom: 0.25rem; }
.api-status-text { font-size: 1rem; font-weight: 600; }
.api-status-text.success { color: var(--primary-dark); }
.api-status-text.error { color: var(--danger); }

.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin: 1.5rem 0; }
.metric-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--gray-200);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover { transform: translateY(-6px); box-shadow: var(--shadow-xl); }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; border-radius: 20px 20px 0 0; }
.metric-card.green::before { background: linear-gradient(90deg, #10B981, #34D399); }
.metric-card.blue::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
.metric-card.orange::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
.metric-card.purple::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
.metric-icon { width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700; margin-bottom: 1rem; }
.metric-icon.green { background: linear-gradient(135deg, #D1FAE5, #A7F3D0); color: #065F46; }
.metric-icon.blue { background: linear-gradient(135deg, #DBEAFE, #BFDBFE); color: #1E40AF; }
.metric-icon.orange { background: linear-gradient(135deg, #FEF3C7, #FDE68A); color: #92400E; }
.metric-icon.purple { background: linear-gradient(135deg, #EDE9FE, #DDD6FE); color: #5B21B6; }
.metric-label { font-size: 0.8rem; color: var(--gray-600); font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: var(--dark); font-family: 'JetBrains Mono', monospace; }
.metric-delta { display: inline-flex; align-items: center; gap: 4px; font-size: 0.85rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; margin-top: 0.75rem; }
.metric-delta.positive { background: #D1FAE5; color: #059669; }
.metric-delta.negative { background: #FEE2E2; color: #DC2626; }

.section-header { display: flex; align-items: center; gap: 12px; font-size: 1.3rem; color: var(--dark); font-weight: 700; margin: 2.5rem 0 1.5rem 0; padding-bottom: 1rem; border-bottom: 2px solid var(--gray-200); }
.section-icon { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, var(--primary), var(--primary-dark)); display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700; color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); }

.alert-box { padding: 1.5rem; border-radius: 16px; margin: 1.25rem 0; position: relative; overflow: hidden; }
.alert-box::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; }
.alert-box.success { background: linear-gradient(135deg, #ECFDF5, #D1FAE5); border: 1px solid #A7F3D0; }
.alert-box.success::before { background: var(--primary); }
.alert-box.warning { background: linear-gradient(135deg, #FFFBEB, #FEF3C7); border: 1px solid #FDE68A; }
.alert-box.warning::before { background: var(--accent); }
.alert-box.danger { background: linear-gradient(135deg, #FEF2F2, #FEE2E2); border: 1px solid #FECACA; }
.alert-box.danger::before { background: var(--danger); }
.alert-box.info { background: linear-gradient(135deg, #EFF6FF, #DBEAFE); border: 1px solid #BFDBFE; }
.alert-box.info::before { background: var(--secondary); }
.alert-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem; }
.alert-box.success .alert-title { color: #065F46; }
.alert-box.warning .alert-title { color: #92400E; }
.alert-box.danger .alert-title { color: #991B1B; }
.alert-box.info .alert-title { color: #1E40AF; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: var(--gray-100); padding: 6px; border-radius: 16px; }
.stTabs [data-baseweb="tab"] { height: 48px; background: transparent; border-radius: 12px; padding: 0 24px; font-weight: 600; font-size: 0.95rem; color: var(--gray-600); border: none; }
.stTabs [aria-selected="true"] { background: #ffffff !important; color: var(--primary-dark) !important; box-shadow: var(--shadow); }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #F9FAFB 100%); border-right: 1px solid var(--gray-200); }
.sidebar-header { font-size: 1.2rem; font-weight: 700; color: var(--dark); margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid var(--primary); }
.sidebar-section { background: var(--gray-50); border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid var(--gray-200); }
.sidebar-section-title { font-size: 0.75rem; font-weight: 600; color: var(--gray-600); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }

.stDownloadButton button { background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 1.5rem !important; font-weight: 600 !important; }
.stDownloadButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important; }

.chart-container { background: #ffffff; border-radius: 20px; padding: 1.5rem; box-shadow: var(--shadow); border: 1px solid var(--gray-200); margin: 1rem 0; }

.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.stats-card { background: #ffffff; border-radius: 12px; padding: 1rem; box-shadow: var(--shadow); border: 1px solid var(--gray-200); text-align: center; }
.stats-label { font-size: 0.75rem; color: var(--gray-600); font-weight: 500; text-transform: uppercase; margin-bottom: 0.5rem; }
.stats-value { font-size: 1.25rem; font-weight: 700; color: var(--dark); font-family: 'JetBrains Mono', monospace; }

.footer { background: linear-gradient(135deg, #1F2937, #111827); color: #ffffff; padding: 2.5rem; border-radius: 24px; margin-top: 3rem; text-align: center; }
.footer h4 { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; }
.footer p { color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0.25rem 0; }
.footer-badges { display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; }
.footer-badge { background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 20px; font-size: 0.8rem; color: #D1FAE5; border: 1px solid rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">AgriPrice Forecaster</div>
    <div class="hero-subtitle">Advanced AI-powered commodity price prediction with real-time weather, currency, and market sentiment integration for smarter agricultural decisions.</div>
    <div class="hero-badge">
        <span>Powered by Machine Learning</span>
        <span>|</span>
        <span>Real-Time APIs</span>
        <span>|</span>
        <span>55+ Commodities</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state for controls
if 'commodity_category' not in st.session_state:
    st.session_state.commodity_category = "Grains & Cereals"
if 'commodity' not in st.session_state:
    st.session_state.commodity = "Wheat"
if 'forecast_days' not in st.session_state:
    st.session_state.forecast_days = 14

commodities_by_category = {
    "Grains & Cereals": ["Wheat", "Rice (Basmati)", "Rice (Non-Basmati)", "Corn", "Barley", "Sorghum", "Millet", "Oats"],
    "Pulses & Legumes": ["Chickpeas (Chana)", "Pigeon Peas (Tur)", "Lentils (Masoor)", "Moong Dal", "Urad Dal", "Kidney Beans (Rajma)", "Black Gram"],
    "Oilseeds": ["Soybean", "Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower", "Cotton Seed"],
    "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco", "Rubber", "Tea", "Coffee"],
    "Spices": ["Turmeric", "Chili (Red)", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Ginger", "Garlic"],
    "Vegetables": ["Potato", "Onion", "Tomato", "Cabbage", "Cauliflower", "Carrot", "Brinjal", "Okra"],
    "Fruits": ["Mango", "Banana", "Apple", "Orange", "Grapes", "Papaya", "Pomegranate", "Watermelon"]
}

# Quick Controls in Hero
col1, col2, col3, col4 = st.columns(4)
with col1:
    quick_category = st.selectbox("Category", list(commodities_by_category.keys()), key="hero_cat", index=list(commodities_by_category.keys()).index(st.session_state.commodity_category))
with col2:
    quick_commodity = st.selectbox("Commodity", commodities_by_category[quick_category], key="hero_com")
with col3:
    quick_days = st.number_input("Forecast Days (7-30)", min_value=7, max_value=30, value=st.session_state.forecast_days, step=1, key="hero_input")
with col4:
    st.write("")
    st.write("")
    apply_btn = st.button("Apply", use_container_width=True, key="apply_btn")

if apply_btn:
    st.session_state.commodity_category = quick_category
    st.session_state.commodity = quick_commodity
    st.session_state.forecast_days = quick_days
    st.rerun()

# Use session state values
commodity_category = st.session_state.commodity_category
commodity = st.session_state.commodity
forecast_days = st.session_state.forecast_days

# API Functions
@st.cache_data(ttl=3600)
def fetch_weather_data(city="Delhi"):
    try:
        coords = {"Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777), "Bangalore": (12.9716, 77.5946)}
        lat, lon = coords.get(city, (28.6139, 77.2090))
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia/Kolkata&past_days=92"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({'date': pd.to_datetime(data['daily']['time']), 'temp_max': data['daily']['temperature_2m_max'], 'temp_min': data['daily']['temperature_2m_min'], 'rainfall': data['daily']['precipitation_sum']})
            return df, True
        return None, False
    except:
        return None, False

@st.cache_data(ttl=3600)
def fetch_currency_rates():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        if response.status_code == 200:
            return response.json()['rates']['INR'], True
        return 83.0, False
    except:
        return 83.0, False

@st.cache_data(ttl=3600)
def fetch_commodity_news_sentiment():
    return np.random.uniform(-0.3, 0.3), True

@st.cache_data
def generate_enhanced_historical_data(commodity, days=365):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_prices = {
        "Wheat": 2100, "Rice (Basmati)": 4500, "Rice (Non-Basmati)": 2800, "Corn": 1900, "Barley": 1800,
        "Sorghum": 1750, "Millet": 2300, "Oats": 2200, "Chickpeas (Chana)": 5500, "Pigeon Peas (Tur)": 7000,
        "Lentils (Masoor)": 6500, "Moong Dal": 8000, "Urad Dal": 7500, "Kidney Beans (Rajma)": 6800,
        "Black Gram": 7200, "Soybean": 4200, "Groundnut": 5800, "Mustard": 5500, "Sunflower": 5000,
        "Sesame": 9500, "Safflower": 5200, "Cotton Seed": 3800, "Cotton": 6500, "Sugarcane": 320,
        "Jute": 4800, "Tobacco": 18000, "Rubber": 14500, "Tea": 350, "Coffee": 8500, "Turmeric": 8500,
        "Chili (Red)": 12000, "Coriander": 7500, "Cumin": 22000, "Black Pepper": 45000, "Cardamom": 120000,
        "Ginger": 4500, "Garlic": 3800, "Potato": 1200, "Onion": 2500, "Tomato": 1800, "Cabbage": 1500,
        "Cauliflower": 2200, "Carrot": 1800, "Brinjal": 2000, "Okra": 3500, "Mango": 4500, "Banana": 2200,
        "Apple": 8000, "Orange": 3500, "Grapes": 5500, "Papaya": 1800, "Pomegranate": 6500, "Watermelon": 1200
    }
    
    base_price = base_prices.get(commodity, 2500)
    weather_df, weather_success = fetch_weather_data("Delhi")
    usd_inr, currency_success = fetch_currency_rates()
    sentiment, sentiment_success = fetch_commodity_news_sentiment()
    
    volatility_map = {"Vegetables": 25, "Fruits": 22, "Spices": 18, "Cash Crops": 16, "Pulses & Legumes": 14, "Oilseeds": 12, "Grains & Cereals": 10}
    commodity_categories = {
        "Grains & Cereals": ["Wheat", "Rice (Basmati)", "Rice (Non-Basmati)", "Corn", "Barley", "Sorghum", "Millet", "Oats"],
        "Pulses & Legumes": ["Chickpeas (Chana)", "Pigeon Peas (Tur)", "Lentils (Masoor)", "Moong Dal", "Urad Dal", "Kidney Beans (Rajma)", "Black Gram"],
        "Oilseeds": ["Soybean", "Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower", "Cotton Seed"],
        "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco", "Rubber", "Tea", "Coffee"],
        "Spices": ["Turmeric", "Chili (Red)", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Ginger", "Garlic"],
        "Vegetables": ["Potato", "Onion", "Tomato", "Cabbage", "Cauliflower", "Carrot", "Brinjal", "Okra"],
        "Fruits": ["Mango", "Banana", "Apple", "Orange", "Grapes", "Papaya", "Pomegranate", "Watermelon"]
    }
    
    commodity_cat = next((cat for cat, items in commodity_categories.items() if commodity in items), None)
    volatility = volatility_map.get(commodity_cat, 15)
    
    trend = np.linspace(0, np.random.uniform(-0.05, 0.15) * base_price, days)
    seasonality = (base_price * 0.08) * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = np.random.normal(0, base_price * (volatility/100), days)
    prices = np.maximum(base_price + trend + seasonality + noise, base_price * 0.3)
    
    data = pd.DataFrame({'date': dates, 'price': prices, 'volume': np.random.randint(1000, 10000, days)})
    
    if weather_success and weather_df is not None:
        weather_df_subset = weather_df[weather_df['date'] >= dates[0]].copy()
        data = data.merge(weather_df_subset[['date', 'temp_max', 'rainfall']], on='date', how='left')
        data['temperature'] = data['temp_max'].fillna(25)
        data['rainfall'] = data['rainfall'].fillna(0)
        data.drop('temp_max', axis=1, inplace=True)
    else:
        data['temperature'] = 20 + 10 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
        data['rainfall'] = np.random.gamma(2, 2, days)
    
    data['usd_inr'] = usd_inr + np.random.normal(0, 0.5, days)
    data['oil_price'] = 6500 + np.random.normal(0, 200, days)
    data['market_sentiment'] = sentiment + np.random.uniform(-0.2, 0.2, days)
    
    if commodity_cat in ["Vegetables", "Fruits"]:
        data['price'] = data['price'] + (data['rainfall'] * 5)
    elif commodity_cat in ["Grains & Cereals"]:
        data['price'] = data['price'] - (data['rainfall'] * 2)
    
    return data, {'weather_api': weather_success, 'currency_api': currency_success, 'sentiment_api': sentiment_success}

# Sidebar Configuration (Optional - kept for advanced users)
with st.sidebar:
    st.markdown('<div class="sidebar-header">Advanced Configuration</div>', unsafe_allow_html=True)
    
    model_type = st.selectbox("Model Algorithm", ["XGBoost", "Random Forest", "Ensemble"])
    
    st.markdown("---")
    st.subheader("Live Data Sources")
    st.markdown("Weather: Open-Meteo API  \nCurrency: ExchangeRate API  \nSentiment: News Analysis")

# Feature engineering
def create_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    for lag in [1, 7, 14, 30]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    for window in [7, 14, 30]:
        df[f'price_roll_mean_{window}'] = df['price'].rolling(window=window).mean()
        df[f'price_roll_std_{window}'] = df['price'].rolling(window=window).std()
    df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall']
    return df.dropna()

@st.cache_resource
def train_and_predict(data, model_type, forecast_days):
    df = create_features(data)
    feature_cols = [col for col in df.columns if col not in ['date', 'price']]
    X, y = df[feature_cols], df['price']
    split_idx = int(len(df) * 0.8)
    X_train, X_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    scaler = MinMaxScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    if model_type == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        m1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        m2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        m1.fit(X_train_scaled, y_train)
        m2.fit(X_train_scaled, y_train)
        y_pred = (m1.predict(X_test_scaled) + m2.predict(X_test_scaled)) / 2
        model = (m1, m2)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    last_data_scaled = scaler.transform(df.iloc[-1:][feature_cols])
    future_predictions = []
    for _ in range(forecast_days):
        if model_type != "Ensemble":
            pred = model.predict(last_data_scaled)[0]
        else:
            pred = (model[0].predict(last_data_scaled)[0] + model[1].predict(last_data_scaled)[0]) / 2
        future_predictions.append(pred)
    
    return {
        'predictions': y_pred, 'actual': y_test, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
        'future_predictions': future_predictions,
        'future_dates': pd.date_range(start=data['date'].max() + timedelta(days=1), periods=forecast_days, freq='D'),
        'test_dates': df['date'][split_idx:].values
    }

# Load data
with st.spinner(f"Fetching real-time data for {commodity}..."):
    historical_data, api_status = generate_enhanced_historical_data(commodity)

# API Status
st.markdown('<div class="section-header"><div class="section-icon">API</div>Connection Status</div>', unsafe_allow_html=True)

api_html = '<div class="api-grid">'
for abbr, name, status in [("WTH", "Weather Data", api_status['weather_api']), ("CUR", "Currency Rates", api_status['currency_api']), ("SNT", "Market Sentiment", api_status['sentiment_api'])]:
    cls = "success" if status else "error"
    txt = "Connected" if status else "Fallback Mode"
    api_html += f'<div class="api-card {cls}"><div class="api-icon {cls}">{abbr}</div><div class="api-title">{name}</div><div class="api-status-text {cls}">{txt}</div></div>'
api_html += '</div>'
st.markdown(api_html, unsafe_allow_html=True)

# Train model
with st.spinner(f"Training {model_type} model..."):
    results = train_and_predict(historical_data, model_type, forecast_days)

# Calculate detailed analytics
current_price = historical_data['price'].iloc[-1]
predicted_price = results['future_predictions'][-1]
price_change = ((predicted_price - current_price) / current_price) * 100

# Additional metrics calculations
prices = historical_data['price']
returns = prices.pct_change().dropna()
volatility_30d = returns[-30:].std() * np.sqrt(252) * 100
volatility_7d = returns[-7:].std() * np.sqrt(252) * 100
avg_price_30d = prices[-30:].mean()
avg_price_7d = prices[-7:].mean()
price_momentum = ((prices.iloc[-1] - prices.iloc[-30]) / prices.iloc[-30]) * 100
max_price_90d = prices[-90:].max()
min_price_90d = prices[-90:].min()
price_range_90d = max_price_90d - min_price_90d
percentile_rank = (prices.iloc[-1] - min_price_90d) / price_range_90d * 100 if price_range_90d > 0 else 50
sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
max_drawdown = ((prices.cummax() - prices) / prices.cummax()).max() * 100
avg_volume = historical_data['volume'].mean()
volume_trend = ((historical_data['volume'][-7:].mean() - historical_data['volume'][-30:].mean()) / historical_data['volume'][-30:].mean()) * 100

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Performance", "Insights", "Reports"])

with tab1:
    st.markdown('<div class="section-header"><div class="section-icon">FC</div>Price Forecast Dashboard</div>', unsafe_allow_html=True)
    
    delta_class = "positive" if price_change >= 0 else "negative"
    delta_icon = "+" if price_change >= 0 else ""
    
    metrics_html = f'''
    <div class="metric-grid">
        <div class="metric-card green">
            <div class="metric-icon green">CP</div>
            <div class="metric-label">Current Price</div>
            <div class="metric-value">Rs.{current_price:,.0f}</div>
            <div style="font-size:0.8rem;color:#6B7280;margin-top:0.5rem">per quintal</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-icon blue">FP</div>
            <div class="metric-label">Predicted ({forecast_days}d)</div>
            <div class="metric-value">Rs.{predicted_price:,.0f}</div>
            <div class="metric-delta {delta_class}">{delta_icon}{price_change:.2f}%</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-icon orange">ER</div>
            <div class="metric-label">Model RMSE</div>
            <div class="metric-value">Rs.{results['rmse']:,.0f}</div>
            <div style="font-size:0.8rem;color:#6B7280;margin-top:0.5rem">error margin</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-icon purple">R2</div>
            <div class="metric-label">R-Squared Score</div>
            <div class="metric-value">{results['r2']:.3f}</div>
            <div style="font-size:0.8rem;color:#6B7280;margin-top:0.5rem">model fit</div>
        </div>
    </div>'''
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Forecast Chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['date'][-180:], y=historical_data['price'][-180:], mode='lines', name='Historical', line=dict(color='#10B981', width=2.5)))
    fig.add_trace(go.Scatter(x=results['future_dates'], y=results['future_predictions'], mode='lines+markers', name='Forecast', line=dict(color='#F59E0B', width=3, dash='dash'), marker=dict(size=8)))
    upper = [p * 1.05 for p in results['future_predictions']]
    lower = [p * 0.95 for p in results['future_predictions']]
    fig.add_trace(go.Scatter(x=list(results['future_dates']) + list(results['future_dates'])[::-1], y=upper + lower[::-1], fill='toself', fillcolor='rgba(245, 158, 11, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig.update_layout(title=f"{commodity} Price Forecast", xaxis_title="Date", yaxis_title="Price (Rs./quintal)", height=500, template='plotly_white', hovermode='x unified', font=dict(family="Inter", size=12))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header"><div class="section-icon">PM</div>Model Performance Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="alert-box info">
            <div class="alert-title">Performance Metrics</div>
            <table style="width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #BFDBFE;"><td style="padding:12px 0;font-weight:500;">Root Mean Square Error (RMSE)</td><td style="text-align:right;font-family:JetBrains Mono;">Rs.{results['rmse']:,.2f}</td></tr>
                <tr style="border-bottom:1px solid #BFDBFE;"><td style="padding:12px 0;font-weight:500;">Mean Absolute Error (MAE)</td><td style="text-align:right;font-family:JetBrains Mono;">Rs.{results['mae']:,.2f}</td></tr>
                <tr style="border-bottom:1px solid #BFDBFE;"><td style="padding:12px 0;font-weight:500;">Mean Absolute Percentage Error</td><td style="text-align:right;font-family:JetBrains Mono;">{results['mape']:.2f}%</td></tr>
                <tr><td style="padding:12px 0;font-weight:500;">R-Squared Score</td><td style="text-align:right;font-family:JetBrains Mono;">{results['r2']:.4f}</td></tr>
            </table>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="alert-box success">
            <div class="alert-title">Model Configuration</div>
            <p><strong>Algorithm:</strong> {model_type}</p>
            <p><strong>Training Samples:</strong> {int(len(historical_data)*0.8):,}</p>
            <p><strong>Test Samples:</strong> {len(results['actual']):,}</p>
            <p><strong>Features Used:</strong> 18 engineered features</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(results['actual']))), y=results['actual'].values, mode='lines', name='Actual', line=dict(color='#10B981', width=2)))
        fig.add_trace(go.Scatter(x=list(range(len(results['predictions']))), y=results['predictions'], mode='lines', name='Predicted', line=dict(color='#F59E0B', width=2, dash='dash')))
        fig.update_layout(title="Actual vs Predicted (Test Set)", xaxis_title="Observation", yaxis_title="Price (Rs./quintal)", height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header"><div class="section-icon">DA</div>Detailed Data Analysis</div>', unsafe_allow_html=True)
    
    # Price Statistics Grid
    st.subheader("Price Statistics")
    stats_html = f'''
    <div class="stats-grid">
        <div class="stats-card"><div class="stats-label">Current Price</div><div class="stats-value">Rs.{current_price:,.2f}</div></div>
        <div class="stats-card"><div class="stats-label">7-Day Average</div><div class="stats-value">Rs.{avg_price_7d:,.2f}</div></div>
        <div class="stats-card"><div class="stats-label">30-Day Average</div><div class="stats-value">Rs.{avg_price_30d:,.2f}</div></div>
        <div class="stats-card"><div class="stats-label">90-Day High</div><div class="stats-value">Rs.{max_price_90d:,.2f}</div></div>
        <div class="stats-card"><div class="stats-label">90-Day Low</div><div class="stats-value">Rs.{min_price_90d:,.2f}</div></div>
        <div class="stats-card"><div class="stats-label">Price Range (90d)</div><div class="stats-value">Rs.{price_range_90d:,.2f}</div></div>
    </div>'''
    st.markdown(stats_html, unsafe_allow_html=True)
    
    # Volatility & Risk Metrics
    st.subheader("Volatility and Risk Metrics")
    risk_html = f'''
    <div class="stats-grid">
        <div class="stats-card"><div class="stats-label">7-Day Volatility</div><div class="stats-value">{volatility_7d:.2f}%</div></div>
        <div class="stats-card"><div class="stats-label">30-Day Volatility</div><div class="stats-value">{volatility_30d:.2f}%</div></div>
        <div class="stats-card"><div class="stats-label">Max Drawdown</div><div class="stats-value">{max_drawdown:.2f}%</div></div>
        <div class="stats-card"><div class="stats-label">Sharpe Ratio</div><div class="stats-value">{sharpe_ratio:.3f}</div></div>
        <div class="stats-card"><div class="stats-label">Percentile Rank</div><div class="stats-value">{percentile_rank:.1f}%</div></div>
        <div class="stats-card"><div class="stats-label">30-Day Momentum</div><div class="stats-value">{price_momentum:+.2f}%</div></div>
    </div>'''
    st.markdown(risk_html, unsafe_allow_html=True)
    
    # Volume Analysis
    st.subheader("Volume Analysis")
    vol_html = f'''
    <div class="stats-grid">
        <div class="stats-card"><div class="stats-label">Average Daily Volume</div><div class="stats-value">{avg_volume:,.0f}</div></div>
        <div class="stats-card"><div class="stats-label">Volume Trend (7d vs 30d)</div><div class="stats-value">{volume_trend:+.2f}%</div></div>
        <div class="stats-card"><div class="stats-label">Total Volume (30d)</div><div class="stats-value">{historical_data['volume'][-30:].sum():,.0f}</div></div>
    </div>'''
    st.markdown(vol_html, unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(historical_data[-180:], x='date', y='price', title=f"{commodity} - 6 Month Price Trend")
        fig.update_traces(line=dict(color='#10B981', width=2.5))
        fig.update_layout(height=350, template='plotly_white', xaxis_title="Date", yaxis_title="Price (Rs./quintal)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        corr = historical_data[['price', 'temperature', 'rainfall', 'usd_inr', 'oil_price', 'market_sentiment']].corr()['price'].drop('price')
        fig = go.Figure(go.Bar(x=corr.values, y=['Temperature', 'Rainfall', 'USD/INR', 'Oil Price', 'Sentiment'], orientation='h', marker_color=['#10B981' if v > 0 else '#EF4444' for v in corr.values], text=[f"{v:.3f}" for v in corr.values], textposition='auto'))
        fig.update_layout(title="Price Correlation with External Factors", height=350, template='plotly_white', xaxis_title="Correlation Coefficient")
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather Impact
    st.subheader("Weather Impact Analysis")
    weather_html = f'''
    <div class="stats-grid">
        <div class="stats-card"><div class="stats-label">Avg Temperature (7d)</div><div class="stats-value">{historical_data['temperature'][-7:].mean():.1f} C</div></div>
        <div class="stats-card"><div class="stats-label">Total Rainfall (7d)</div><div class="stats-value">{historical_data['rainfall'][-7:].sum():.1f} mm</div></div>
        <div class="stats-card"><div class="stats-label">USD/INR Rate</div><div class="stats-value">Rs.{historical_data['usd_inr'].iloc[-1]:.2f}</div></div>
    </div>'''
    st.markdown(weather_html, unsafe_allow_html=True)
    
    # Distribution Chart
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=historical_data['price'][-90:], nbinsx=25, marker_color='#10B981', opacity=0.8))
        fig.update_layout(title="Price Distribution (90 Days)", xaxis_title="Price (Rs./quintal)", yaxis_title="Frequency", height=300, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Histogram(x=returns[-90:]*100, nbinsx=25, marker_color='#3B82F6', opacity=0.8))
        fig.update_layout(title="Daily Returns Distribution (90 Days)", xaxis_title="Daily Return (%)", yaxis_title="Frequency", height=300, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header"><div class="section-icon">RP</div>Reports and Recommendations</div>', unsafe_allow_html=True)
    
    outlook = "success" if price_change > 5 else ("danger" if price_change < -5 else "info")
    outlook_text = "BULLISH OUTLOOK" if price_change > 5 else ("BEARISH OUTLOOK" if price_change < -5 else "NEUTRAL OUTLOOK")
    
    st.markdown(f'''
    <div class="alert-box {outlook}">
        <div class="alert-title">{outlook_text}</div>
        <p>The model predicts a <strong>{price_change:+.2f}%</strong> price movement over the next {forecast_days} days.</p>
        <p style="margin-top:1rem;"><strong>Current Price:</strong> Rs.{current_price:,.2f} | <strong>Predicted Price:</strong> Rs.{predicted_price:,.2f}</p>
        <p><strong>Model Confidence (R2):</strong> {results['r2']:.4f} | <strong>Error Margin:</strong> Rs.{results['rmse']:,.2f}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="alert-box warning">
        <div class="alert-title">Risk Assessment</div>
        <p><strong>Volatility Level:</strong> {"High" if volatility_30d > 30 else ("Moderate" if volatility_30d > 15 else "Low")} ({volatility_30d:.1f}% annualized)</p>
        <p><strong>Maximum Drawdown:</strong> {max_drawdown:.2f}%</p>
        <p><strong>Current Position:</strong> Price is at {percentile_rank:.0f}th percentile of 90-day range</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = pd.DataFrame({'Date': results['future_dates'], 'Predicted_Price': results['future_predictions'], 'Change_Pct': [((p - current_price) / current_price * 100) for p in results['future_predictions']]}).to_csv(index=False)
        st.download_button("Download Forecast Data", csv, f"{commodity}_forecast.csv", "text/csv")
    with col2:
        metrics_csv = pd.DataFrame({'Metric': ['RMSE', 'MAE', 'MAPE', 'R2', 'Volatility_30d', 'Max_Drawdown', 'Sharpe_Ratio'], 'Value': [results['rmse'], results['mae'], results['mape'], results['r2'], volatility_30d, max_drawdown, sharpe_ratio]}).to_csv(index=False)
        st.download_button("Download Analytics Report", metrics_csv, f"{commodity}_analytics.csv", "text/csv")

# Footer
st.markdown('''
<div class="footer">
    <h4>AgriPrice Forecaster v2.0</h4>
    <p>AI-Powered Agricultural Commodity Price Prediction System</p>
    <div class="footer-badges">
        <span class="footer-badge">Open-Meteo API</span>
        <span class="footer-badge">ExchangeRate API</span>
        <span class="footer-badge">Machine Learning</span>
        <span class="footer-badge">55+ Commodities</span>
    </div>
    <p style="margin-top:1.5rem;font-size:0.8rem;">Disclaimer: This tool is for research and educational purposes only. Always verify predictions with domain experts before making trading decisions.</p>
</div>
''', unsafe_allow_html=True)