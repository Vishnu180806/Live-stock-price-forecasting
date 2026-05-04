# Live Stock Price Forecasting Dashboard 📈

A real-time stock price forecasting dashboard built with Streamlit, using ARIMA and Linear Regression models to predict future price trends.

## Features
- **Real-time Data**: Fetches live stock data using `yfinance`.
- **Forecasting Models**: 
    - **ARIMA**: For time-series analysis and forecasting.
    - **Linear Regression**: For trend-based predictions.
- **Interactive Visualizations**: Interactive charts powered by Plotly.
- **Adjustable Parameters**: Change stock symbols, data periods, and forecast duration on the fly.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vishnu180806/Live-stock-price-forecasting.git
   cd Live-stock-price-forecasting
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run python.py
   ```

## Usage
- Enter a stock symbol (e.g., `AAPL`, `GOOGL`, `TSLA`).
- Select the historical data period.
- Use the slider to set the number of forecast days.
- View the metrics, model performance, and interactive forecast charts.

## License
MIT
