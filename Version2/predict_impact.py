import pickle
import numpy as np
import os

SAVE_DIR = r"C:\Users\Ahmad Shah\OneDrive - ahmadsyedshah123@gmail.com\OneDrive\General\Obsidian\General\Projects\StockBot\Version2"
MODEL_FILE = os.path.join(SAVE_DIR, "trained_model.pkl")

# Load model and label encoder
with open(MODEL_FILE, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    label_encoder = data["label_encoder"]

def predict_price_change(confidence, sentiment, previous_change, moving_avg):
    # Encode sentiment
    sentiment_encoded = label_encoder.transform([sentiment])[0]
    features = np.array([confidence, sentiment_encoded, previous_change, moving_avg]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
