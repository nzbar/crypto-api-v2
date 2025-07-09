import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

def load_features(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # ✅ تحويل dict إلى list إذا كانت الأسطر عبارة عن قواميس
    if isinstance(data[0], dict):
        keys = sorted(data[0].keys())  # ثبات الترتيب
        data = [[row[k] for k in keys] for row in data]

    arr = np.array(data, dtype=np.float32)
    print(f"✅ Loaded data shape: {arr.shape}")  # (total_days, num_features)
    return arr

def create_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+forecast_horizon])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dense(output_shape[0] * output_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main(args):
    # إعدادات
    sequence_length = 60
    forecast_horizon = 10

    # تحميل البيانات
    data = load_features(args.features_file)

    # تطبيع البيانات
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # إعداد تسلسل التدريب
    X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)
    print(f"✅ Training X shape: {X.shape}, y shape: {y.shape}")  # (samples, 60, 246) و (samples, 10, 246)

    # بناء النموذج
    num_features = X.shape[2]
    model = build_lstm_model((sequence_length, num_features), (forecast_horizon, num_features))

    # تدريب
    model.fit(X, y.reshape(X.shape[0], -1), epochs=20, batch_size=32)

    # حفظ النموذج
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.model_name}.h5")
    print(f"✅ Model saved as models/{args.model_name}.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True, help="Path to features.json")
    parser.add_argument("--model_name", type=str, default="lstm_forecast", help="Name to save the trained model")
    args = parser.parse_args()

    main(args)
