import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Live Stock Forecast", layout="wide")

st.title("📈 Live Stock Price Forecasting Dashboard")
st.markdown("Real-time stock data with ARIMA & Linear Regression forecasts. Updates automatically.")

# Sidebar
st.sidebar.header("Live Inputs")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
days = st.sidebar.slider("Forecast Days", 5, 60, 30)

@st.cache_data(ttl=300)  # Cache 5 min for live feel
def fetch_live_data(symbol, period):
    try:
        data = yf.download(symbol, period=period, progress=False)["Close"]
        return data.dropna().squeeze()
    except:
        return pd.Series()

if st.sidebar.button("🔄 Refresh Live Data", type="primary"):
    fetch_live_data.clear()

data = fetch_live_data(symbol, period)

if data is None or data.empty:
    if symbol.strip() == "":
        st.warning("Enter a stock symbol to view live data.")
    else:
        st.warning(f"No data found for '{symbol}'. If this is a non-US stock, make sure to add the exchange suffix! (e.g., '{symbol}.NS' for NSE India, '.BO' for BSE).")
elif len(data) < 100:
    st.error("Insufficient data. Try longer period.")
else:
    st.success(f"✅ Live data loaded: {len(data):,} days for {symbol} (last: ${data.iloc[-1]:.2f})")

    # Split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${data.iloc[-1]:.2f}")
    
    with col2:
        change = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100
        st.metric("1-Day Change", f"{change:+.2f}%")
    
    with col3:
        vola = data.pct_change().std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{vola:.1f}%")

    # ARIMA
    with st.expander("ARIMA Model (Time Series)", expanded=True):
        try:
            model = ARIMA(train, order=(1,1,1)).fit()
            forecast_arima = model.forecast(steps=days)
            pred_test = model.forecast(steps=len(test))
            rmse_arima = np.sqrt(mean_squared_error(test, pred_test))
            st.info(f"RMSE on test: {rmse_arima:.2f}")
        except Exception as e:
            st.error(f"ARIMA error: {e}")
            forecast_arima = np.full(days, data.iloc[-1])

    # Regression
    with st.expander("Linear Regression (Trend)", expanded=True):
        df = pd.DataFrame({"price": data})
        lags = 10
        for i in range(1, lags+1):
            df[f"lag_{i}"] = df["price"].shift(i)
        df.dropna(inplace=True)
        
        X = df[[f"lag_{i}" for i in range(1, lags+1)]][:train_size]
        y = df["price"][:train_size]
        reg = LinearRegression().fit(X, y)
        
        last_lags = df[[f"lag_{i}" for i in range(1, lags+1)]].iloc[-1:].values
        forecast_reg = []
        current_lags = last_lags.copy()
        for _ in range(days):
            pred = reg.predict(current_lags)[0]
            forecast_reg.append(pred)
            current_lags = np.roll(current_lags, -1)
            current_lags[0, -1] = pred
        
        forecast_reg = np.array(forecast_reg)
        rmse_reg = np.sqrt(mean_squared_error(df["price"][train_size:train_size+min(20,len(test))], 
                                              reg.predict(df[[f"lag_{i}" for i in range(1, lags+1)]][train_size:train_size+min(20,len(test))])))
        st.info(f"RMSE on test: {rmse_reg:.2f}")

    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, name="Live Historical", line=dict(color='blue')))
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days, freq='B')
    
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_arima, name="ARIMA Forecast", 
                             line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_reg, name="Regression Forecast", 
                             line=dict(color='green', dash='dot')))
    
    fig.update_layout(title=f"{symbol} Live Price Forecast", xaxis_title="Date", yaxis_title="Price ($)",
                      height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    forecast_arima = np.array(forecast_arima)
    forecast_reg = np.array(forecast_reg)
    
    fc_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "ARIMA": np.round(forecast_arima, 2),
        "Regression": np.round(forecast_reg, 2),
        "ARIMA Change %": np.diff(forecast_arima, prepend=forecast_arima[0]) / forecast_arima[0] * 100,
        "Reg Change %": np.diff(forecast_reg, prepend=forecast_reg[0]) / forecast_reg[0] * 100
    })
    st.dataframe(fc_df, use_container_width=True)

st.markdown("---")
st.caption("🔄 Data refreshes on button click. Forecasts are for educational purposes only. Not financial advice.")
