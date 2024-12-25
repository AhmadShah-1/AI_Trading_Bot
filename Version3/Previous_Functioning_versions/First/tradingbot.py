# tradingbot.py

import os
import pickle
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

from finbert_utils import estimate_sentiment
from config import API_KEY, API_SECRET, BASE_URL  # Import from config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILENAME = "trained_model.pkl"

# Check if model file exists
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Missing {MODEL_FILENAME}. Please train your model first.")

# Load the pre-trained model
with open(MODEL_FILENAME, "rb") as f:
    TRAINED_MODEL = pickle.load(f)

SYMBOL = "SPY"
CASH_AT_RISK = 0.5
SLEEPTIME = "24H"

class MLTrader(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.sleeptime = SLEEPTIME
        self.cash_at_risk = CASH_AT_RISK
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.model = TRAINED_MODEL

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = int(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        current_date = self.get_datetime().date()
        date_str = current_date.strftime('%Y-%m-%d')

        # 1) Fetch today's news from Alpaca
        news_items = self.api.get_news(
            symbol=self.symbol,
            start=date_str,
            end=date_str,
            limit=50
        )
        headlines = [item.headline for item in news_items]
        if headlines:
            sent_label, sent_conf = estimate_sentiment(headlines)
        else:
            sent_label, sent_conf = ("neutral", 0.0)

        # 2) Fetch today's price data from yfinance
        price_data = yf.download(
            self.symbol,
            start=current_date,
            end=current_date + timedelta(days=1)
        ).dropna()

        if price_data.empty:
            logger.warning(f"No price data available for {current_date}. Skipping iteration.")
            return

        # Convert price_data to a standard index
        price_data.reset_index(drop=True, inplace=True)

        # Extract the first row
        open_price = price_data.iloc[0]["Open"]
        high_price = price_data.iloc[0]["High"]
        low_price = price_data.iloc[0]["Low"]
        close_price = price_data.iloc[0]["Close"]
        volume = price_data.iloc[0]["Volume"]

        # 3) Feature engineering
        daily_return = (close_price - open_price) / open_price
        daily_volatility = (high_price - low_price) / open_price

        # Convert sentiment label to numeric
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        sent_label_encoded = mapping.get(sent_label, 1)

        # 4) Build a DataFrame with the same columns as in training
        features_dict = {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Volume": volume,
            "daily_return": daily_return,
            "daily_volatility": daily_volatility,
            "sentiment_label_encoded": sent_label_encoded,
            "sentiment_confidence": sent_conf
        }
        X_today = pd.DataFrame([features_dict], columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "daily_return",
            "daily_volatility",
            "sentiment_label_encoded",
            "sentiment_confidence"
        ])

        # 5) Predict up (1) or down (0) for the next day
        prediction = self.model.predict(X_today)[0]

        # 6) Check current position
        position = self.get_position(self.symbol)

        if prediction == 1:
            # Buy signal
            if position is None or position.quantity == 0:
                if cash > (last_price * quantity):
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="market"
                    )
                    self.submit_order(order)
                    logger.info(f"BUY {quantity} shares of {self.symbol} at {last_price}")
        else:
            # Sell signal
            if position is not None and position.quantity > 0:
                self.sell_all()
                logger.info(f"SELL all shares of {self.symbol}")

# Dates for backtesting
TRAINING_END_DATE = datetime(2023, 3, 1)
TESTING_START_DATE = TRAINING_END_DATE + timedelta(days=1)
TESTING_END_DATE = datetime(2024, 5, 1)

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
        parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK}
    )

    # Backtest
    strategy.backtest(
        YahooDataBacktesting,
        TESTING_START_DATE,
        TESTING_END_DATE,
        parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK}
    )

    # Uncomment to run live (paper or real) after confirming everything
    """
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
    """
