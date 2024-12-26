# tradingbot.py

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- Lumibot / Alpaca imports ---
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api.rest import REST

# --- TensorFlow / Keras imports ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Joblib for loading the scaler/encoder ---
import joblib

# --- Custom modules ---
from finbert_utils import estimate_sentiment
from config import API_KEY, API_SECRET, BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1) Load the Keras LSTM Model and Scaler/Encoder
# ---------------------------------------------------------------------
MODEL_FILENAME = "optimized_hybrid_stock_price_model.keras"
SCALER_FILENAME = "scaler.pkl"
ENCODER_FILENAME = "encoder.pkl"

if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(
        f"Missing {MODEL_FILENAME}. Please train your model first."
    )

if not os.path.exists(SCALER_FILENAME):
    raise FileNotFoundError(
        f"Missing {SCALER_FILENAME}. You must save the MinMaxScaler from train_model.py."
    )

if not os.path.exists(ENCODER_FILENAME):
    raise FileNotFoundError(
        f"Missing {ENCODER_FILENAME}. You must save the OneHotEncoder from train_model.py."
    )

TRAINED_MODEL = load_model(MODEL_FILENAME)
SCALER = joblib.load(SCALER_FILENAME)
ENCODER = joblib.load(ENCODER_FILENAME)

SYMBOL = "SPY"
SLEEPTIME = "24H"

# ---------------------------------------------------------------------
# 2) Helper: Build the 60-Day Sequence for LSTM
# ---------------------------------------------------------------------
def build_lstm_features_for_today(current_date, sentiment_label, sentiment_confidence):
    """
    Fetch the last 60 trading days of SPY data (including 'current_date'),
    compute daily_return, daily_volatility, and prepare the exact same
    feature vector we used in training. Then scale it with SCALER, and
    add one-hot encoding of the sentiment_label for the *last day*.

    Returns X of shape (1, 60, num_features) or (None, None) if insufficient data.

    numeric_features we expect:
      "Adj Close", "Close", "High", "Low", "Open", "Volume",
      "daily_return", "daily_volatility"
    + "sentiment_confidence"
    + 3 columns from the OneHotEncoder => total 12 columns
    """

    lookback_days = 60
    start_date = current_date - timedelta(days=lookback_days * 3)  # buffer for weekends/holidays
    end_date = current_date + timedelta(days=1)

    # Download data
    df_price = yf.download(SYMBOL, start=start_date, end=end_date, progress=False)
    df_price.dropna(inplace=True)
    df_price.reset_index(inplace=True)
    df_price.rename(columns={"Date": "date"}, inplace=True)
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.date
    df_price.sort_values("date", inplace=True)
    df_price.reset_index(drop=True, inplace=True)

    # Ensure "Adj Close" exists in case yfinance omits it
    if "Adj Close" not in df_price.columns:
        df_price["Adj Close"] = df_price["Close"]

    # Compute daily_return, daily_volatility
    df_price["daily_return"] = (df_price["Close"] - df_price["Open"]) / df_price["Open"]
    df_price["daily_volatility"] = (df_price["High"] - df_price["Low"]) / df_price["Open"]

    # Keep only rows up to current_date
    df_price = df_price[df_price["date"] <= current_date]
    if len(df_price) < lookback_days:
        logger.warning(
            f"Not enough data to form a 60-day sequence. Found only {len(df_price)} days."
        )
        return None, None

    # Only the last 60 days
    df_60 = df_price.iloc[-lookback_days:].copy()

    # We'll store "neutral" for each day except the last day (today)
    sentiments = ["neutral"] * lookback_days
    sentiments[-1] = sentiment_label  # today's sentiment
    confidences = [0.0] * lookback_days
    confidences[-1] = sentiment_confidence

    # Build numeric features for each of the 60 days
    numeric_features = [
        "Adj Close", "Close", "High", "Low", "Open", "Volume",
        "daily_return", "daily_volatility"
    ]

    data_60 = []
    for i in range(lookback_days):
        row = df_60.iloc[i]
        row_numeric = [
            row["Adj Close"],
            row["Close"],
            row["High"],
            row["Low"],
            row["Open"],
            row["Volume"],
            row["daily_return"],
            row["daily_volatility"],
            confidences[i],   # sentiment_confidence
        ]
        # One-hot encode the sentiment_label
        temp_df = pd.DataFrame({"sentiment_label": [sentiments[i]]})
        # Because we used OneHotEncoder(..., sparse=False), transform() returns a numpy array
        # so we do NOT call .toarray()
        one_hot_array = ENCODER.transform(temp_df[["sentiment_label"]])[0]

        # Combine numeric + sentiment one-hot
        row_full = np.hstack([row_numeric, one_hot_array])
        data_60.append(row_full)

    data_60 = np.array(data_60)  # shape (60, 12)

    # Scale using SCALER
    data_60_scaled = SCALER.transform(data_60)

    # LSTM expects shape (1, 60, num_features)
    X = np.expand_dims(data_60_scaled, axis=0)
    return X, df_60

# ---------------------------------------------------------------------
# 3) Strategy: MLTrader
# ---------------------------------------------------------------------
class MLTrader(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.sleeptime = SLEEPTIME
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.model = TRAINED_MODEL  # The loaded LSTM model

    def get_portfolio_value(self):
        """
        Sum up cash + the market value of all positions.
        """
        total_value = self.get_cash()
        positions = self.get_positions()
        for pos in positions:
            sym = pos.symbol
            last_price = self.get_last_price(sym)
            total_value += (pos.quantity * last_price)
        return total_value

    def on_trading_iteration(self):
        current_date = self.get_datetime().date()
        logger.info(f"=== Trading iteration for {current_date} ===")

        # 1) Fetch today's news, estimate sentiment
        date_str = current_date.strftime('%Y-%m-%d')
        news_items = self.api.get_news(
            symbol=self.symbol,
            start=date_str,
            end=date_str,
            limit=50
        )
        headlines = [item.headline for item in news_items]
        logger.debug(f"Fetched {len(headlines)} headlines for {current_date}: {headlines}")

        if headlines:
            sent_label, sent_conf = estimate_sentiment(headlines)
        else:
            sent_label, sent_conf = ("neutral", 0.0)

        logger.debug(f"Sentiment => label={sent_label}, confidence={sent_conf:.3f}")

        # 2) Build LSTM feature sequence
        X_today, df_60 = build_lstm_features_for_today(
            current_date, sent_label, sent_conf
        )
        if X_today is None:
            # Not enough data for a 60-day sequence
            logger.warning("Skipping iteration due to insufficient data.")
            return

        # The last row in df_60 should be today's row (or last trading day).
        last_close_price = float(df_60.iloc[-1]["Close"])

        # 3) Model prediction:
        # The model was trained to predict the next day's Close in scaled form
        pred_scaled = self.model.predict(X_today)  # shape: (1,1)
        pred_scaled_close = pred_scaled[0][0]      # scalar
        logger.debug(f"Raw model prediction (scaled) = {pred_scaled_close:.4f}")

        # We need to invert the scaling for the 'Close' column.
        # In our original feature arrangement, 'Close' is index=1 in the array
        #   [Adj Close=0, Close=1, High=2, Low=3, Open=4, Volume=5, daily_return=6,
        #    daily_volatility=7, sentiment_conf=8, one-hot=9,10,11]
        dummy_row = np.zeros((1, X_today.shape[2]))
        dummy_row[0, 1] = pred_scaled_close
        inv_row = SCALER.inverse_transform(dummy_row)
        predicted_close = inv_row[0, 1]
        logger.info(f"Predicted Close for next day (unscaled) = {predicted_close:.4f}")

        # 4) Decide how to interpret the prediction as "prob_up"
        # A simple approach: if predicted_close > today's close => prob_up=1, else 0
        if predicted_close > last_close_price:
            prob_up = 1.0
        else:
            prob_up = 0.0

        logger.info(f"prob_up={prob_up:.2f} for next day after {current_date}")

        # 5) fraction_invested = prob_up
        fraction_invested = prob_up
        logger.debug(f"Fraction_invested={fraction_invested:.2f}")

        # 6) Check portfolio value, decide how many shares to hold
        portfolio_value = self.get_portfolio_value()
        logger.debug(f"Current portfolio_value={portfolio_value:.2f}")

        if portfolio_value <= 0:
            logger.warning("Portfolio value <= 0. Skipping iteration.")
            return

        last_price = self.get_last_price(self.symbol)
        desired_shares = int((portfolio_value * fraction_invested) / last_price)
        logger.debug(f"Desired_shares={desired_shares}")

        current_position = self.get_position(self.symbol)
        current_shares = current_position.quantity if current_position else 0
        difference = desired_shares - current_shares
        logger.debug(f"Difference in shares={difference}")

        # 7) Place an order
        if difference > 0:
            # Buy
            cash_available = self.get_cash()
            cost_estimate = difference * last_price
            if cost_estimate <= cash_available:
                order = self.create_order(self.symbol, difference, "buy", "market")
                self.submit_order(order)
                logger.info(
                    f"BUY {difference} shares => total {desired_shares}, "
                    f"prob_up={prob_up:.2f}"
                )
            else:
                logger.warning(
                    f"Not enough cash to buy {difference} shares. "
                    f"Needed={cost_estimate:.2f}, have={cash_available:.2f}"
                )
        elif difference < 0:
            # Sell
            shares_to_sell = abs(difference)
            order = self.create_order(self.symbol, shares_to_sell, "sell", "market")
            self.submit_order(order)
            logger.info(
                f"SELL {shares_to_sell} shares => total {desired_shares}, "
                f"prob_up={prob_up:.2f}"
            )
        else:
            logger.info(
                f"No trade needed. fraction_invested={fraction_invested:.2f}, "
                f"current_shares={current_shares}, prob_up={prob_up:.2f}"
            )

# ---------------------------------------------------------------------
# 4) Backtesting Setup
# ---------------------------------------------------------------------
TRAINING_END_DATE = datetime(2023, 3, 1)
TESTING_START_DATE = TRAINING_END_DATE + timedelta(days=1)
TESTING_END_DATE = datetime(2024, 4, 1)


# Only run main code in __main__ to avoid Windows multiprocessing issues
if __name__ == "__main__":
    ALPACA_CREDS = {
        "API_KEY": API_KEY,
        "API_SECRET": API_SECRET,
        "PAPER": True,
        "BASE_URL": BASE_URL
    }

    broker = Alpaca(ALPACA_CREDS)

    strategy = MLTrader(
        name="MLTrader",
        broker=broker,
        parameters={"symbol": SYMBOL}
    )

    # Run backtest
    strategy.backtest(
        YahooDataBacktesting,
        TESTING_START_DATE,
        TESTING_END_DATE,
        parameters={"symbol": SYMBOL}
    )

    # If you want to run live, uncomment:
    """
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
    """

