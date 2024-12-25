import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import os
import re

SAVE_DIR = r"C:\Users\Ahmad Shah\OneDrive - ahmadsyedshah123@gmail.com\OneDrive\General\Obsidian\General\Projects\StockBot\Version2"
DATA_FILE = os.path.join(SAVE_DIR, "sentiment_data.csv")
MODEL_FILE = os.path.join(SAVE_DIR, "trained_model.pkl")

# Load historical data
df = pd.read_csv(DATA_FILE)

# Ensure the 'date' column is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract numerical values from the "confidence" column
def extract_confidence(conf):
    if isinstance(conf, str) and "tensor" in conf:
        match = re.search(r"tensor\(([\d.]+)", conf)
        return float(match.group(1)) if match else None
    return conf

df["confidence"] = df["confidence"].apply(extract_confidence)

# Encode sentiment to numerical values
label_encoder = LabelEncoder()
df["sentiment_encoded"] = label_encoder.fit_transform(df["sentiment"])

# Filter data for training (10 months ago to 5 months ago)
today = pd.Timestamp.now()
train_start_date = today - pd.Timedelta(days=300)  # 10 months ago
train_end_date = today - pd.Timedelta(days=150)    # 5 months ago

# Ensure filtering works with datetime
train_data = df[(df["date"] >= train_start_date) & (df["date"] < train_end_date)].copy()

# Create trend-based features
train_data["previous_price_change"] = train_data["price_change"].shift(1)
train_data["moving_avg_7"] = train_data["price_change"].rolling(window=7).mean()
train_data.fillna(0, inplace=True)

# Prepare features and target
X = train_data[["confidence", "sentiment_encoded", "previous_price_change", "moving_avg_7"]]
y = (train_data["price_change"] > 0).astype(int)  # 1 for increase, 0 for decrease

# Train-test split within the training period
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and label encoder
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"model": model, "label_encoder": label_encoder}, f)
