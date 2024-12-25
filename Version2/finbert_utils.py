from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
import yfinance as yf
import os

# Alpaca API Credentials
ALPACA_API_KEY = "PKQQ047X0DYICHUNMY6D"
ALPACA_API_SECRET = "mN18V0GvRncs2KVEiF2dROuKJM1eRxLAKXPtXZmC"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
alpaca_api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, base_url=ALPACA_BASE_URL)

# Directory to save the sentiment data
SAVE_DIR = r"C:\Users\Ahmad Shah\OneDrive - ahmadsyedshah123@gmail.com\OneDrive\General\Obsidian\General\Projects\StockBot\Version2"
OUTPUT_FILE = os.path.join(SAVE_DIR, "sentiment_data.csv")

# FinBERT model and tokenizer setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def fetch_alpaca_prices(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical stock prices from Alpaca using the IEX feed.
    Args:
        symbol (str): Stock symbol (e.g., "SPY").
        start_date (datetime): Start date for data.
        end_date (datetime): End date for data.
    Returns:
        pd.DataFrame: A DataFrame containing historical prices and price changes.
    """
    price_data = alpaca_api.get_bars(
        symbol=symbol,
        timeframe="1Day",
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        feed="iex"  # Use IEX data feed
    ).df

    # Calculate price change
    price_data["price_change"] = (price_data["close"] - price_data["open"]) / price_data["open"]
    price_data.reset_index(inplace=True)
    price_data.rename(columns={"timestamp": "date"}, inplace=True)
    price_data["date"] = pd.to_datetime(price_data["date"]).dt.date
    return price_data

def fetch_yahoo_prices(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical stock prices from Yahoo Finance.
    Args:
        symbol (str): Stock symbol (e.g., "SPY").
        start_date (datetime): Start date for data.
        end_date (datetime): End date for data.
    Returns:
        pd.DataFrame: A DataFrame containing historical prices and price changes.
    """
    price_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    price_data.reset_index(inplace=True)
    price_data["price_change"] = (price_data["Close"] - price_data["Open"]) / price_data["Open"]
    price_data.rename(columns={"Date": "date"}, inplace=True)
    return price_data

def fetch_alpaca_news(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch news headlines from Alpaca within the specified date range for a given symbol.
    Args:
        symbol (str): Stock symbol (e.g., "SPY").
        start_date (datetime): Start date for news fetching.
        end_date (datetime): End date for news fetching.
    Returns:
        pd.DataFrame: A DataFrame containing headlines and their publication dates.
    """
    news_data = []
    for single_date in pd.date_range(start=start_date, end=end_date):
        start_str = single_date.strftime('%Y-%m-%d')
        end_str = (single_date + timedelta(days=1)).strftime('%Y-%m-%d')
        news = alpaca_api.get_news(symbol=symbol, start=start_str, end=end_str)
        for item in news:
            news_data.append({
                "date": single_date.date(),
                "headline": item.__dict__["_raw"]["headline"]
            })

    return pd.DataFrame(news_data)

def estimate_sentiment(headlines):
    """
    Estimate sentiment and confidence for a given list of headlines.
    Args:
        headlines (list): List of news headlines (strings).
    Returns:
        Tuple (float, str): Confidence score (0-1) and sentiment label (positive/negative/neutral).
    """
    if not headlines:
        return 0.0, "neutral"  # Default for empty input

    # Tokenize and process headlines
    tokens = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get model predictions
    logits = model(**tokens).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Aggregate results
    aggregated_probs = torch.mean(probabilities, dim=0)
    confidence = aggregated_probs.max().item()  # Highest probability as confidence score
    sentiment = labels[aggregated_probs.argmax().item()]  # Label with highest probability

    return confidence, sentiment

def fetch_and_prepare_data(symbol: str, start_date: datetime, end_date: datetime, use_yahoo=False) -> pd.DataFrame:
    """
    Fetch and prepare data by combining stock prices and news headlines with sentiment analysis.
    Args:
        symbol (str): Stock symbol (e.g., "SPY").
        start_date (datetime): Start date for data.
        end_date (datetime): End date for data.
        use_yahoo (bool): Use Yahoo Finance for price data if True, otherwise use Alpaca.
    Returns:
        pd.DataFrame: A DataFrame with sentiment analysis and price change data.
    """
    if use_yahoo:
        price_data = fetch_yahoo_prices(symbol, start_date, end_date)
    else:
        price_data = fetch_alpaca_prices(symbol, start_date, end_date)

    news_data = fetch_alpaca_news(symbol, start_date, end_date)

    # Merge news and price data
    combined_df = pd.merge(news_data, price_data, on="date", how="inner")

    # Perform sentiment analysis
    sentiments = []
    confidences = []
    for _, row in combined_df.iterrows():
        confidence, sentiment = estimate_sentiment([row["headline"]])
        confidences.append(confidence)
        sentiments.append(sentiment)

    combined_df["confidence"] = confidences
    combined_df["sentiment"] = sentiments
    return combined_df


if __name__ == "__main__":
    start = datetime.now() - timedelta(days=150)  # Fetch last 150 days of data
    end = datetime.now()
    symbol = "SPY"

    # Fetch and analyze data
    sentiment_data = fetch_and_prepare_data(symbol, start, end, use_yahoo=False)

    # Save to CSV
    sentiment_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Sentiment data saved to {OUTPUT_FILE}")
