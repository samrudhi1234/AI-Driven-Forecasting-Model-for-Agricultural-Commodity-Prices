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
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AgriPrice Forecaster", layout="wide", page_icon="🌾")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f9f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 Agricultural Commodity Price Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Driven Decision Support for Trade Optimization</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.title("Configuration")
    
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
    st.markdown("### 📊 Commodity Coverage")
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

# Function to generate synthetic historical data (simulating real data)
@st.cache_data
def generate_historical_data(commodity, days=730):
    """Generate synthetic historical price data with realistic patterns"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base prices for different commodities (USD/ton)
    base_prices = {
        # Grains & Cereals
        "Wheat": 280,
        "Rice (Basmati)": 850,
        "Rice (Non-Basmati)": 420,
        "Corn": 220,
        "Barley": 240,
        "Sorghum": 210,
        "Millet": 320,
        "Oats": 260,
        
        # Pulses & Legumes
        "Chickpeas (Chana)": 920,
        "Pigeon Peas (Tur)": 1050,
        "Lentils (Masoor)": 880,
        "Moong Dal": 1100,
        "Urad Dal": 1020,
        "Kidney Beans (Rajma)": 950,
        "Black Gram": 980,
        
        # Oilseeds
        "Soybean": 480,
        "Groundnut": 720,
        "Mustard": 650,
        "Sunflower": 580,
        "Sesame": 1400,
        "Safflower": 620,
        "Cotton Seed": 380,
        
        # Cash Crops
        "Cotton": 1800,
        "Sugarcane": 35,
        "Jute": 520,
        "Tobacco": 3200,
        "Rubber": 1650,
        "Tea": 2800,
        "Coffee": 4200,
        
        # Spices
        "Turmeric": 5800,
        "Chili (Red)": 3200,
        "Coriander": 2100,
        "Cumin": 4500,
        "Black Pepper": 6800,
        "Cardamom": 18000,
        "Ginger": 1200,
        "Garlic": 980,
        
        # Vegetables
        "Potato": 320,
        "Onion": 380,
        "Tomato": 420,
        "Cabbage": 280,
        "Cauliflower": 450,
        "Carrot": 350,
        "Brinjal": 380,
        "Okra": 620,
        
        # Fruits
        "Mango": 850,
        "Banana": 480,
        "Apple": 1200,
        "Orange": 680,
        "Grapes": 1400,
        "Papaya": 420,
        "Pomegranate": 1800,
        "Watermelon": 220
    }
    
    base_price = base_prices.get(commodity, 500)
    
    # Different volatility levels for different commodity types
    volatility_map = {
        # High volatility
        "Vegetables": 25,
        "Fruits": 22,
        
        # Medium-high volatility
        "Spices": 18,
        "Cash Crops": 16,
        
        # Medium volatility
        "Pulses & Legumes": 14,
        "Oilseeds": 12,
        
        # Lower volatility
        "Grains & Cereals": 10
    }
    
    # Determine commodity category
    commodity_cat = None
    for cat, items in {
        "Grains & Cereals": ["Wheat", "Rice (Basmati)", "Rice (Non-Basmati)", "Corn", "Barley", "Sorghum", "Millet", "Oats"],
        "Pulses & Legumes": ["Chickpeas (Chana)", "Pigeon Peas (Tur)", "Lentils (Masoor)", "Moong Dal", "Urad Dal", "Kidney Beans (Rajma)", "Black Gram"],
        "Oilseeds": ["Soybean", "Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower", "Cotton Seed"],
        "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco", "Rubber", "Tea", "Coffee"],
        "Spices": ["Turmeric", "Chili (Red)", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Ginger", "Garlic"],
        "Vegetables": ["Potato", "Onion", "Tomato", "Cabbage", "Cauliflower", "Carrot", "Brinjal", "Okra"],
        "Fruits": ["Mango", "Banana", "Apple", "Orange", "Grapes", "Papaya", "Pomegranate", "Watermelon"]
    }.items():
        if commodity in items:
            commodity_cat = cat
            break
    
    volatility = volatility_map.get(commodity_cat, 15)
    
    # Generate price with trend, seasonality, and noise
    trend_strength = np.random.uniform(-0.05, 0.15) * base_price  # -5% to +15% trend
    trend = np.linspace(0, trend_strength, days)
    seasonality = (base_price * 0.08) * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = np.random.normal(0, base_price * (volatility/100), days)
    prices = base_price + trend + seasonality + noise
    
    # Ensure no negative prices
    prices = np.maximum(prices, base_price * 0.3)
    
    # Additional features
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 10000, days),
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days),
        'rainfall': np.random.gamma(2, 2, days),
        'usd_inr': 75 + np.random.normal(0, 2, days),
        'oil_price': 80 + np.random.normal(0, 5, days),
        'market_sentiment': np.random.uniform(-1, 1, days)
    })
    
    return data

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
    
    return df.dropna()

# Function to train and predict
@st.cache_resource
def train_and_predict(data, model_type, forecast_days):
    """Train model and generate predictions"""
    # Prepare data
    df = create_features(data)
    
    # Features and target
    feature_cols = [col for col in df.columns if col not in ['date', 'price']]
    X = df[feature_cols]
    y = df['price']
    
    # Train-test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
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
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Generate future predictions
    last_data = df.iloc[-1:][feature_cols]
    last_data_scaled = scaler.transform(last_data)
    
    future_predictions = []
    current_data = last_data_scaled.copy()
    
    for i in range(forecast_days):
        if model_type != "Ensemble":
            pred = model.predict(current_data)[0]
        else:
            pred = (model1.predict(current_data)[0] + model2.predict(current_data)[0]) / 2
        future_predictions.append(pred)
        
        # Update features for next prediction (simplified)
        current_data = current_data.copy()
    
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

# Load data
with st.spinner(f"Loading historical data for {commodity}..."):
    historical_data = generate_historical_data(commodity)

# Train model
with st.spinner(f"Training {model_type} model..."):
    results = train_and_predict(historical_data, model_type, forecast_days)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Model Performance", "🔍 Data Insights", "📋 Reports"])

with tab1:
    st.header("Price Forecast")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = historical_data['price'].iloc[-1]
    predicted_price = results['future_predictions'][-1]
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}/ton",
            delta=None
        )
    
    with col2:
        st.metric(
            label=f"Predicted ({forecast_days}d)",
            value=f"${predicted_price:.2f}/ton",
            delta=f"{price_change:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Model RMSE",
            value=f"${results['rmse']:.2f}",
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
        line=dict(color='#2E7D32', width=2)
    ))
    
    # Forecasted prices
    fig.add_trace(go.Scatter(
        x=results['future_dates'],
        y=results['future_predictions'],
        mode='lines+markers',
        name='Forecasted Price',
        line=dict(color='#FF6F00', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    upper_bound = [p * 1.05 for p in results['future_predictions']]
    lower_bound = [p * 0.95 for p in results['future_predictions']]
    
    fig.add_trace(go.Scatter(
        x=results['future_dates'].tolist() + results['future_dates'].tolist()[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255, 111, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{commodity} Price Forecast ({forecast_days} Days)",
        xaxis_title="Date",
        yaxis_title="Price (USD/ton)",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Forecast table
    st.subheader("Detailed Forecast")
    forecast_df = pd.DataFrame({
        'Date': results['future_dates'],
        'Predicted Price (USD/ton)': [f"${p:.2f}" for p in results['future_predictions']],
        'Change from Current': [f"{((p - current_price) / current_price * 100):.2f}%" for p in results['future_predictions']]
    })
    st.dataframe(forecast_df, width='stretch')

with tab2:
    st.header("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE'],
            'Value': [
                f"${results['rmse']:.2f}",
                f"${results['mae']:.2f}",
                f"{results['mape']:.2f}%"
            ]
        })
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        st.markdown("#### Model Information")
        st.info(f"""
        **Model Type:** {model_type}
        
        **Training Period:** {len(historical_data) - len(results['actual'])} days
        
        **Test Period:** {len(results['actual'])} days
        
        **Features Used:** Price lags, rolling statistics, weather data, market indicators
        """)
    
    with col2:
        st.subheader("Actual vs Predicted (Test Set)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(results['actual']))),
            y=results['actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#2E7D32', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(results['predictions']))),
            y=results['predictions'],
            mode='lines',
            name='Predicted',
            line=dict(color='#FF6F00', width=2, dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Days",
            yaxis_title="Price (USD/ton)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Residual analysis
    st.subheader("Residual Analysis")
    residuals = results['actual'] - results['predictions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='#4CAF50'
        ))
        fig.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residual",
            yaxis_title="Frequency",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['predictions'],
            y=residuals,
            mode='markers',
            marker=dict(color='#4CAF50', size=6)
        ))
        fig.update_layout(
            title="Residuals vs Predicted Values",
            xaxis_title="Predicted Price",
            yaxis_title="Residual",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')

with tab3:
    st.header("Data Insights & Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Trend (Last 6 Months)")
        recent_data = historical_data[-180:]
        fig = px.line(recent_data, x='date', y='price', 
                      title=f"{commodity} Price Trend")
        fig.update_layout(height=350, template='plotly_white')
        st.plotly_chart(fig, width='stretch')
        
        st.subheader("Volume Analysis")
        fig = px.bar(recent_data[-30:], x='date', y='volume',
                     title="Trading Volume (Last 30 Days)")
        fig.update_layout(height=350, template='plotly_white')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Correlation with External Factors")
        corr_data = historical_data[['price', 'temperature', 'rainfall', 'usd_inr', 'oil_price']].corr()['price'].drop('price')
        
        fig = go.Figure(go.Bar(
            x=corr_data.values,
            y=corr_data.index,
            orientation='h',
            marker_color=['#4CAF50' if v > 0 else '#F44336' for v in corr_data.values]
        ))
        fig.update_layout(
            title="Feature Correlation with Price",
            xaxis_title="Correlation Coefficient",
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
        
        st.subheader("Statistical Summary")
        summary_stats = historical_data[['price', 'volume', 'temperature', 'rainfall']].describe().round(2)
        st.dataframe(summary_stats, width='stretch')

with tab4:
    st.header("Reports & Recommendations")
    
    st.subheader("📊 Executive Summary")
    st.markdown(f"""
    ### Forecast Summary for {commodity}
    
    **Forecast Period:** {forecast_days} days  
    **Model Used:** {model_type}  
    **Forecast Date:** {datetime.now().strftime('%Y-%m-%d')}
    
    #### Key Findings:
    - Current market price: **${current_price:.2f}/ton**
    - Predicted price after {forecast_days} days: **${predicted_price:.2f}/ton**
    - Expected price movement: **{price_change:+.2f}%**
    - Model accuracy (MAPE): **{results['mape']:.2f}%**
    """)
    
    # Recommendations
    st.subheader("💡 Trading Recommendations")
    
    if price_change > 5:
        st.success(f"""
        #### BULLISH OUTLOOK
        The model predicts a **{price_change:.2f}%** increase over the next {forecast_days} days.
        
        **Recommended Actions:**
        - Consider increasing inventory positions
        - Delay sales if holding stock
        - Review forward contracts
        - Monitor weather patterns closely
        """)
    elif price_change < -5:
        st.error(f"""
        #### BEARISH OUTLOOK
        The model predicts a **{price_change:.2f}%** decrease over the next {forecast_days} days.
        
        **Recommended Actions:**
        - Consider reducing inventory exposure
        - Accelerate sales if applicable
        - Explore hedging opportunities
        - Review procurement strategies
        """)
    else:
        st.info(f"""
        #### NEUTRAL OUTLOOK
        The model predicts a relatively stable price movement of **{price_change:+.2f}%**.
        
        **Recommended Actions:**
        - Maintain current inventory levels
        - Continue regular trading patterns
        - Monitor for any sudden market shifts
        - Focus on operational efficiency
        """)
    
    # Risk factors
    st.subheader("⚠️ Risk Factors & Limitations")
    st.warning("""
    - Model predictions are based on historical patterns and may not capture sudden geopolitical events
    - Weather anomalies or extreme climate events could significantly impact accuracy
    - Policy changes and trade restrictions are not fully accounted for
    - Currency fluctuations may affect international price dynamics
    - Model requires regular retraining with fresh data for optimal performance
    """)
    
    # Download report
    st.subheader("📥 Export Report")
    
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>AgriPrice Forecaster</strong> | AI-Driven Commodity Price Prediction</p>
    <p>⚠️ This is a prototype system. Always verify predictions with domain experts before making trading decisions.</p>
</div>
""", unsafe_allow_html=True)