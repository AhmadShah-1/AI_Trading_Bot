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
from config import API_KEY, API_SECRET, BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILENAME = "trained_model.pkl"

if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Missing {MODEL_FILENAME}. Please train your model first.")

with open(MODEL_FILENAME, "rb") as f:
    TRAINED_MODEL = pickle.load(f)

SYMBOL = "SPY"
SLEEPTIME = "24H"

class MLTrader(Strategy):
    def initialize(self):
        self.symbol = SYMBOL
        self.sleeptime = SLEEPTIME
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.model = TRAINED_MODEL

    def get_portfolio_value(self):
        """
        Sum up cash + the market value of all positions.
        In Lumibot backtesting, self.get_positions() is typically a list of positions.
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

        # 1) Fetch today's news
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

        # 2) Fetch price data
        price_data = yf.download(
            self.symbol,
            start=current_date,
            end=current_date + timedelta(days=1)
        ).dropna()

        if price_data.empty:
            logger.warning(f"No price data for {current_date}. Skipping iteration.")
            return

        price_data.reset_index(drop=True, inplace=True)

        # -- CRITICAL: cast these to float --
        open_price = float(price_data["Open"].iloc[0])
        high_price = float(price_data["High"].iloc[0])
        low_price = float(price_data["Low"].iloc[0])
        close_price = float(price_data["Close"].iloc[0])
        volume = float(price_data["Volume"].iloc[0])

        logger.debug(
            f"Price data => O={open_price:.2f}, H={high_price:.2f}, "
            f"L={low_price:.2f}, C={close_price:.2f}, V={volume}"
        )

        # 3) Feature engineering
        daily_return = (close_price - open_price) / open_price
        daily_volatility = (high_price - low_price) / open_price

        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        sent_label_encoded = mapping.get(sent_label, 1)

        # 4) Build feature row
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

        X_today = pd.DataFrame([features_dict], columns=features_dict.keys()).astype(float)

        # 5) Predict probability
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_today)[0]
            prob_up = proba[1]
        else:
            pred = self.model.predict(X_today)[0]
            prob_up = 1.0 if pred == 1 else 0.0

        logger.info(f"Model prob_up={prob_up:.3f} for {current_date}")

        # 6) fraction_invested = prob_up (0.0 to 1.0)
        fraction_invested = max(0.0, min(prob_up, 1.0))
        logger.debug(f"Fraction_invested={fraction_invested:.3f}")

        # 7) portfolio_value
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
        logger.debug(f"Current_shares={current_shares}")

        difference = desired_shares - current_shares
        logger.debug(f"Difference in shares={difference}")

        # 8) place order
        if difference > 0:
            cash_available = self.get_cash()
            cost_estimate = difference * last_price
            if cost_estimate <= cash_available:
                order = self.create_order(self.symbol, difference, "buy", "market")
                self.submit_order(order)
                logger.info(
                    f"BUY {difference} => total {desired_shares}, "
                    f"prob_up={prob_up:.3f}, fraction={fraction_invested:.3f}"
                )
            else:
                logger.warning(
                    f"Not enough cash to buy {difference} shares. "
                    f"Needed={cost_estimate:.2f}, have={cash_available:.2f}"
                )
        elif difference < 0:
            shares_to_sell = abs(difference)
            order = self.create_order(self.symbol, shares_to_sell, "sell", "market")
            self.submit_order(order)
            logger.info(
                f"SELL {shares_to_sell} => total {desired_shares}, "
                f"prob_up={prob_up:.3f}, fraction={fraction_invested:.3f}"
            )
        else:
            logger.info(
                f"No trade. fraction_invested={fraction_invested:.3f}, "
                f"current_shares={current_shares}, prob_up={prob_up:.3f}"
            )


# Backtest date range
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

    strategy = MLTrader(
        name="MLTrader",
        broker=broker,
        parameters={"symbol": SYMBOL}
    )

    strategy.backtest(
        YahooDataBacktesting,
        TESTING_START_DATE,
        TESTING_END_DATE,
        parameters={"symbol": SYMBOL}
    )

    # Uncomment if you want to run live:
    """
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
    """
