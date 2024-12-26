import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

CSV_FILENAME = "stock_data_with_news_20216_20231.csv"
MODEL_FILENAME = "optimized_hybrid_stock_price_model.keras"

def preprocess_data(df, time_steps=60):
    """
    Preprocess data for training, including feature scaling, encoding, and lagging.
    """
    # Encode sentiment_label
    encoder = OneHotEncoder(sparse_output=False)
    sentiment_encoded = encoder.fit_transform(df[["sentiment_label"]])

    # Combine numeric and encoded sentiment features
    numeric_features = [
        "Adj Close", "Close", "High", "Low", "Open", "Volume",
        "daily_return", "daily_volatility", "sentiment_confidence"
    ]
    numeric_data = df[numeric_features].values
    full_data = np.hstack((numeric_data, sentiment_encoded))

    # Normalize features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(full_data)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")

    # Prepare LSTM sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 1])  # Target is 'Close'
    return np.array(X), np.array(y), scaler


def build_hybrid_lstm_model(input_shape):
    """
    Build a hybrid LSTM model for stock price prediction.
    """
    input_layer = Input(shape=input_shape)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    lstm_layer = LSTM(64, return_sequences=False)(lstm_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)

    dense_layer = Dense(64, activation="relu")(lstm_layer)
    output_layer = Dense(1, activation="linear")(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mse"])
    return model

def main():
    # Load the dataset
    df = pd.read_csv(CSV_FILENAME)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Preprocess the data
    time_steps = 60
    X, y, scaler = preprocess_data(df, time_steps)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Build and train the model
    model = build_hybrid_lstm_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

    # Save the model
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")

    # Evaluate the model
    test_loss, test_mse = model.evaluate(X_test, y_test)
    print(f"Test MSE: {test_mse:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Predict and visualize results
    y_pred = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(np.hstack((np.zeros((len(y_test), X_test.shape[2] - 1)), y_test.reshape(-1, 1))))[:, -1]
    y_pred_actual = scaler.inverse_transform(np.hstack((np.zeros((len(y_pred), X_test.shape[2] - 1)), y_pred)))[:, -1]

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label="Actual Prices", color="blue")
    plt.plot(y_pred_actual, label="Predicted Prices", color="red")
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
