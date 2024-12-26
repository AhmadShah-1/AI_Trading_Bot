# tradingbot.py

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api.rest import REST

import tensorflow as tf
from tensorflow.keras.models import load_model

import joblib

from finbert_utils import estimate_sentiment
from config import API_KEY, API_SECRET, BASE_URL

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load models and tools
MODEL_FILENAME = "optimized_hybrid_stock_price_model.keras"
SCALER_FILENAME = "scaler.pkl"
ENCODER_FILENAME = "encoder.pkl"

TRAINED_MODEL = load_model(MODEL_FILENAME)
SCALER = joblib.load(SCALER_FILENAME)
ENCODER = joblib.load(ENCODER_FILENAME)

SYMBOL = "SPY"
SLEEPTIME = "24H"

def build_lstm_features_for_today(current_date, sentiment_label, sentiment_confidence):
    lookback_days = 60
    start_date = current_date - timedelta(days=lookback_days * 3)
    end_date = current_date + timedelta(days=1)

    df_price = yf.download(SYMBOL, start=start_date, end=end_date, progress=False)
    if df_price.empty:
        logger.error("No data retrieved from Yahoo Finance.")
        return None, None

    df_price.reset_index(inplace=True)
    if "Adj Close" not in df_price.columns:
        df_price["Adj Close"] = df_price["Close"]

    df_price.rename(columns={"Date": "date"}, inplace=True)
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.date
    df_price.sort_values("date", inplace=True)
    df_price.reset_index(drop=True, inplace=True)

    # Add derived columns
    df_price["daily_return"] = df_price["Close"].pct_change()
    df_price["daily_volatility"] = df_price["Close"].rolling(window=5).std()

    if len(df_price) < lookback_days:
        logger.error("Not enough data for the required lookback period.")
        return None, None

    df_60 = df_price.iloc[-lookback_days:].copy()
    sentiments = ["neutral"] * lookback_days
    sentiments[-1] = sentiment_label
    confidences = [0.0] * lookback_days
    confidences[-1] = sentiment_confidence


    numeric_cols = [
        "Adj Close", "Close", "High", "Low", "Open",
        "Volume", "daily_return", "daily_volatility"
    ]

    # Validate numeric_cols
    missing_cols = [col for col in numeric_cols if col not in df_60.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None, None

    data_60 = []
    for i in range(lookback_days):
        row_numeric = df_60.loc[df_60.index[i], numeric_cols].to_numpy(dtype=float).tolist()
        row_numeric.append(confidences[i])
        temp_df = pd.DataFrame({"sentiment_label": [sentiments[i]]})
        one_hot_array = ENCODER.transform(temp_df[["sentiment_label"]])[0]
        row_full = np.concatenate([row_numeric, one_hot_array])
        data_60.append(row_full)

    data_60 = np.array(data_60, dtype=float)
    data_60_scaled = SCALER.transform(data_60)
    X = np.expand_dims(data_60_scaled, axis=0)
    return X, df_60

class MLTrader(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.sleeptime = SLEEPTIME
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.model = TRAINED_MODEL

    def on_trading_iteration(self):
        current_date = self.get_datetime().date()
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            news_items = self.api.get_news(symbol=self.symbol, start=date_str, end=date_str, limit=50)
            headlines = [item.headline for item in news_items]
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return

        # Estimate sentiment (FinBERT)
        sent_label, sent_conf = estimate_sentiment(headlines) if headlines else ("neutral", 0.0)

        # Build LSTM features for today's date
        X_today, df_60 = build_lstm_features_for_today(current_date, sent_label, sent_conf)
        if X_today is None:
            return

        # Get the last close price from our 60-day data
        last_close_price = float(df_60["Close"].iloc[-1])

        # Predict scaled close
        pred_scaled = self.model.predict(X_today)
        pred_scaled_close = pred_scaled[0][0]

        # "Invert" the scale for the 'Close' column
        dummy_row = np.zeros((1, X_today.shape[2]))
        # Index=1 in the numeric features array is "Close"
        dummy_row[0, 1] = pred_scaled_close
        inv_row = SCALER.inverse_transform(dummy_row)
        predicted_close = inv_row[0, 1]

        # Decide whether to go long, short, or do nothing
        difference = predicted_close - last_close_price
        threshold = 0.00025 * last_close_price  # 0.5% threshold

        if difference > threshold:
            # Strong buy signal
            prob_up = 1.0
        elif difference < -threshold:
            # Strong sell (short) signal
            prob_up = -1.0
        else:
            # Neutral - do nothing
            prob_up = 0.0

        # Compute how many shares we want to hold
        portfolio_value = self.get_portfolio_value()
        last_price = self.get_last_price(self.symbol)

        # Go "all in" in either direction
        desired_shares = int((portfolio_value / last_price) * prob_up)

        current_position = self.get_position(self.symbol)
        current_shares = current_position.quantity if current_position else 0

        difference_shares = desired_shares - current_shares

        if difference_shares > 0:
            order_side = "buy"
        elif difference_shares < 0:
            order_side = "sell"
        else:
            # Already in desired position
            order_side = None

        # Submit the order if there's a difference
        if order_side is not None and difference_shares != 0:
            logger.info(
                f"Placing {order_side} order for {abs(difference_shares)} shares "
                f"(predicted_close={predicted_close:.2f}, last_close_price={last_close_price:.2f})"
            )
            order = self.create_order(
                self.symbol,
                abs(difference_shares),
                side=order_side
            )
            self.broker.submit_order(order)


TRAINING_END_DATE = datetime(2023, 3, 1)
TESTING_START_DATE = TRAINING_END_DATE + timedelta(days=1)
TESTING_END_DATE = datetime(2024, 4, 1)

if __name__ == "__main__":
    ALPACA_CREDS = {
        "API_KEY": API_KEY,
        "API_SECRET": API_SECRET,
        "PAPER": True,
        "BASE_URL": BASE_URL
    }

    broker = Alpaca(ALPACA_CREDS)
    strategy = MLTrader(name="MLTrader", broker=broker, parameters={"symbol": SYMBOL})

    strategy.backtest(
        YahooDataBacktesting,
        TESTING_START_DATE,
        TESTING_END_DATE,
        parameters={"symbol": SYMBOL}
    )
