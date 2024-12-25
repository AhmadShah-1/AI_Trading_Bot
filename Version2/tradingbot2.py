from datetime import datetime, timedelta
import pandas as pd
import os
from predict_impact import predict_price_change
from finbert_utils import estimate_sentiment

SAVE_DIR = r"C:\Users\Ahmad Shah\OneDrive - ahmadsyedshah123@gmail.com\OneDrive\General\Obsidian\General\Projects\StockBot\Version2"
DATA_FILE = os.path.join(SAVE_DIR, "sentiment_data.csv")

# Load historical data
df = pd.read_csv(DATA_FILE)

# Ensure the 'date' column is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Define periods
today = pd.Timestamp.now()
test_start_date = today - pd.Timedelta(days=150)  # 5 months ago

# Filter test data (last 5 months)
test_data = df[df["date"] >= test_start_date]


def test_strategy(test_data):
    """Simulate the strategy on unknown data."""
    for _, row in test_data.iterrows():
        confidence, sentiment = estimate_sentiment([row["headline"]])

        # Calculate trend features
        previous_change = test_data["price_change"].shift(1) if "price_change" in test_data.columns else 0
        moving_avg = test_data["price_change"].rolling(window=7).mean() if "price_change" in test_data.columns else 0

        prediction = predict_price_change(confidence, sentiment, previous_change, moving_avg)

        action = "Buy" if prediction == 1 else "Sell"
        print(f"Prediction: {action}, Sentiment: {sentiment}, Confidence: {confidence}, Last Price: {row['close']}")


if __name__ == "__main__":
    test_strategy(test_data)
