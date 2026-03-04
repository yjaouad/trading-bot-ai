import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Try to import transformers for FinBERT, fallback if fails
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from sklearn.linear_model import LinearRegression
import torch

# --- CONFIGURATION ---
st.set_page_config(page_title="Trading Bot AI - Trend & Sentiment", layout="wide")

# Assets mapping
ASSETS = {
    "Pétrole (WTI)": "CL=F",
    "Gaz Naturel": "NG=F",
    "EUR/USD": "EURUSD=X"
}

# Load Sentiment Model (FinBERT)
@st.cache_resource
def load_sentiment_model():
    if not HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None

sentiment_pipe = load_sentiment_model()

# --- DATA FUNCTIONS ---
def get_historical_data(symbol, period="1mo", interval="1h"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df

def add_technical_indicators(df):
    if df.empty:
        return df
    
    # SMA 20, 50
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD 12, 26, 9
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    
    return df

def get_sentiment(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    if not news:
        return "Neutre", 0.5, []
    
    headlines = [item.get('title', item.get('headline', '')) for item in news[:5]]
    headlines = [h for h in headlines if h] # Filter out empty
    
    if not headlines:
        return "Neutre", 0.5, []
    
    avg_score = 0.5 # Default
    if sentiment_pipe:
        try:
            results = sentiment_pipe(headlines)
            sentiment_map = {"positive": 1, "neutral": 0.5, "negative": 0}
            scores = [sentiment_map[res['label']] for res in results]
            avg_score = sum(scores) / len(scores)
            used_finbert = True
        except Exception:
            used_finbert = False
    else:
        used_finbert = False

    if not used_finbert:
        # Fallback: keyword based
        pos_words = ['up', 'high', 'buy', 'growth', 'gain', 'positive', 'bullish', 'increase']
        neg_words = ['down', 'low', 'sell', 'drop', 'loss', 'negative', 'bearish', 'decrease']
        scores = []
        for h in headlines:
            h_lower = h.lower()
            p = sum(1 for w in pos_words if w in h_lower)
            n = sum(1 for w in neg_words if w in h_lower)
            scores.append(0.5 + (p * 0.1) - (n * 0.1))
        avg_score = sum(scores) / len(scores)
    
    final_label = "Positif" if avg_score > 0.55 else "Négatif" if avg_score < 0.45 else "Neutre"
    return final_label, avg_score, news[:5]

def predict_future_prices(df, days_ahead=[7, 15, 30]):
    if df.empty or len(df) < 50:
        return {}
    
    # Prepare data for Linear Regression
    # We use index as X (time) and Close as y (price)
    df_reset = df.reset_index()
    X = np.array(df_reset.index).reshape(-1, 1)
    y = df_reset['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = {}
    last_index = X[-1][0]
    last_price = y[-1]
    
    # Assuming 1h interval for forecasting simplicity in calculation
    # or roughly estimating based on recent slope
    for days in days_ahead:
        # If interval is 1h, 1 day = 24 points. If 1d, 1 day = 1 point.
        # Let's detect frequency or use a robust factor
        # For simplicity, we'll assume the model slope applies per day if we normalize X
        # But here X is just raw index. Let's calculate points to add.
        # We'll use the last 100 points to capture recent trend better than the whole history
        
        recent_X = X[-100:]
        recent_y = y[-100:]
        model_recent = LinearRegression()
        model_recent.fit(recent_X, recent_y)
        
        # Estimate points per day based on time delta between last two rows
        time_delta = (df.index[-1] - df.index[-2]).total_seconds() / 3600 # hours
        points_per_day = 24 / time_delta if time_delta > 0 else 1
        
        future_index = last_index + (days * points_per_day)
        pred_price = model_recent.predict([[future_index]])[0]
        
        change_pct = ((pred_price - last_price) / last_price) * 100
        predictions[days] = {"price": pred_price, "change": change_pct}
        
    return predictions

# --- UI COMPONENTS ---
def draw_forecast_chart(df, predictions, asset_name):
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historique', line=dict(color='white', width=2)))
    
    # Predictions
    future_dates = [df.index[-1] + timedelta(days=d) for d in predictions.keys()]
    future_prices = [p['price'] for p in predictions.values()]
    
    # Add a line connecting last price to predictions
    all_dates = [df.index[-1]] + future_dates
    all_prices = [df['Close'].iloc[-1]] + future_prices
    
    fig.add_trace(go.Scatter(x=all_dates, y=all_prices, 
                             name='Prévision (Tendance)', 
                             line=dict(color='cyan', width=2, dash='dot'),
                             mode='lines+markers'))
    
    fig.update_layout(title=f"Prévision de Tendance : {asset_name}", yaxis_title="Prix", template="plotly_dark", height=400)
    return fig

def draw_chart(df, asset_name):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Prix'))

    # SMA 20 & 50
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA 20'))
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='blue', width=1), name='SMA 50'))

    fig.update_layout(title=f"Graphique {asset_name}", yaxis_title="Prix", template="plotly_dark", height=600)
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# --- MAIN APP ---
def main():
    st.title("🤖 Trading Bot AI: Tendances & Sentiments")
    st.markdown("Analyse en temps réel du Pétrole, Gaz et EUR/USD.")

    # Sidebar
    st.sidebar.header("Paramètres")
    asset_choice = st.sidebar.selectbox("Choisir un actif", list(ASSETS.keys()))
    period = st.sidebar.selectbox("Période", ["1d", "5d", "1mo", "6mo", "1y"], index=2)
    interval = st.sidebar.selectbox("Intervalle", ["15m", "30m", "1h", "1d"], index=2)

    symbol = ASSETS[asset_choice]

    # Loading data
    with st.spinner(f"Chargement des données pour {asset_choice}..."):
        df = get_historical_data(symbol, period, interval)
        df = add_technical_indicators(df)
        sentiment_label, sentiment_score, news_items = get_sentiment(symbol)

    # Layout: Top columns for summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        change = current_price - df['Close'].iloc[-2]
        st.metric("Prix Actuel", f"{current_price:.4f}", f"{change:.4f}")

    with col2:
        # Simple trend detection logic
        last_sma20 = df['SMA_20'].iloc[-1]
        last_sma50 = df['SMA_50'].iloc[-1]
        trend = "Haussier" if last_sma20 > last_sma50 else "Baissier"
        st.metric("Tendance Technique", trend)

    with col3:
        st.metric("Sentiment du Marché", sentiment_label)

    # Predictions Metrics
    st.subheader("🔮 Prévisions Futuristes (Basées sur la Tendance)")
    preds = predict_future_prices(df)
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        if 7 in preds:
            st.metric("Dans 7 jours", f"{preds[7]['price']:.4f}", f"{preds[7]['change']:.2f}%")
    with p_col2:
        if 15 in preds:
            st.metric("Dans 15 jours", f"{preds[15]['price']:.4f}", f"{preds[15]['change']:.2f}%")
    with p_col3:
        if 30 in preds:
            st.metric("Dans 30 jours", f"{preds[30]['price']:.4f}", f"{preds[30]['change']:.2f}%")

    # Main Charts
    tabs = st.tabs(["Graphique Technique", "Prévisions"])
    with tabs[0]:
        st.plotly_chart(draw_chart(df, asset_choice), use_container_width=True)
    with tabs[1]:
        if preds:
            st.plotly_chart(draw_forecast_chart(df, preds, asset_choice), use_container_width=True)
            st.info("Note: Les prévisions sont basées sur une régression linéaire des 100 derniers points de données. Elles indiquent la direction de la tendance et non une certitude absolue.")
        else:
            st.warning("Pas assez de données pour générer des prévisions.")

    # Detailed Analysis
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Indicateurs Techniques")
        st.write(f"- **RSI**: {df['RSI_14'].iloc[-1]:.2f}")
        macd_val = df['MACD_12_26_9'].iloc[-1]
        macd_signal = df['MACDs_12_26_9'].iloc[-1]
        st.write(f"- **MACD**: {macd_val:.4f} (Signal: {macd_signal:.4f})")
        
        rsi_val = df['RSI_14'].iloc[-1]
        if rsi_val > 70:
            st.warning("Surachat (RSI > 70)")
        elif rsi_val < 30:
            st.success("Survente (RSI < 30)")
        else:
            st.info("RSI Neutre")

    with c2:
        st.subheader("Dernières Actualités & Sentiment")
        for item in news_items:
            title = item.get('title', item.get('headline', 'No Title'))
            link = item.get('link', '#')
            st.markdown(f"- [{title}]({link})")
        
        st.progress(sentiment_score)
        st.caption("Score de sentiment (0=Négatif, 1=Positif)")

    # Final Prediction
    st.divider()
    st.header("🔮 Prédiction Globale")
    
    # Combined Logic (Very basic example)
    tech_score = 1 if trend == "Haussier" else 0
    total_score = (tech_score + sentiment_score) / 2
    
    if total_score > 0.7:
        st.success("🔥 SIGNAL D'ACHAT FORT")
    elif total_score > 0.55:
        st.success("📈 SIGNAL D'ACHAT")
    elif total_score < 0.3:
        st.error("📉 SIGNAL DE VENTE FORT")
    elif total_score < 0.45:
        st.error("📉 SIGNAL DE VENTE")
    else:
        st.warning("⚖️ ATTENTE / NEUTRE")

if __name__ == "__main__":
    main()
