# create_csv.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST

from finbert_utils import estimate_sentiment
from config import API_KEY, API_SECRET, BASE_URL  # <--- your config

SYMBOL = "SPY"
START_DATE = datetime(2021, 6, 1)
END_DATE = datetime(2023, 1, 1)
CSV_FILENAME = "stock_data_with_news_20216_20231.csv"

def main():
    # Create Alpaca REST client
    api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)

    print(f"Downloading price data for {SYMBOL} from {START_DATE.date()} to {END_DATE.date()} via yfinance...")
    price_data = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    price_data.dropna(inplace=True)
    print(f"Downloaded {len(price_data)} rows of price data.\n")

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.droplevel(1)

    df = price_data.copy()
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)

    print("Columns after flattening multiindex and renaming:")
    print(df.columns)
    print("First few rows:")
    print(df.head(3))
    print()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["daily_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["daily_volatility"] = (df["High"] - df["Low"]) / df["Open"]

    df["sentiment_label"] = "neutral"
    df["sentiment_confidence"] = 0.0
    df["news_headlines"] = ""

    total_rows = len(df)
    print(f"Processing news for {total_rows} trading days...")

    for i in range(total_rows):
        current_date = df.iloc[i]["date"]
        start_str = current_date.strftime("%Y-%m-%d")
        end_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"[{i+1}/{total_rows}] Fetching news for {current_date}... ", end="")
        try:
            news_items = api.get_news(
                symbol=SYMBOL,
                start=start_str,
                end=end_str,
                limit=50
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        headlines = [item.headline for item in news_items]
        print(f"Found {len(headlines)} headline(s).", end="")

        if headlines:
            sent_label, sent_conf = estimate_sentiment(headlines)
            df.at[i, "news_headlines"] = " || ".join(headlines)
            df.at[i, "sentiment_label"] = sent_label
            df.at[i, "sentiment_confidence"] = sent_conf
            print(f"  Sentiment={sent_label}, Confidence={sent_conf:.3f}")
        else:
            print("  No headlines.")

    df.to_csv(CSV_FILENAME, index=False)
    print(f"\nSaved CSV to {CSV_FILENAME}")

if __name__ == "__main__":
    main()
