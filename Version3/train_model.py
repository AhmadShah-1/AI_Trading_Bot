# train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

CSV_FILENAME = "stock_data_with_news_20216_20231.csv"
MODEL_FILENAME = "trained_model.pkl"

def encode_sentiment_label(label: str) -> int:
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    return mapping.get(label, 1)

def main():
    # 1) Load CSV
    df = pd.read_csv(CSV_FILENAME)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # 2) Next day return as target
    df["next_day_return"] = df["daily_return"].shift(-1)
    df.dropna(inplace=True)

    # 3) Classification: 1 if next_day_return > 0
    df["target"] = (df["next_day_return"] > 0).astype(int)

    # 4) Encode sentiment
    df["sentiment_label_encoded"] = df["sentiment_label"].apply(encode_sentiment_label)

    # 5) Choose features
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "daily_return",
        "daily_volatility",
        "sentiment_label_encoded",
        "sentiment_confidence",
    ]
    X = df[features].copy()
    y = df["target"].copy()

    # 6) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 7) Train RandomForest
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 8) Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 9) Save model
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_FILENAME}")

if __name__ == "__main__":
    main()
