import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess(data_path, industry_id, lookback=24, pred_horizon=24):
    df = pd.read_csv(data_path)
    df = df[df['industry_id'] == industry_id].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    co2_values = df['co2_kg'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    co2_scaled = scaler.fit_transform(co2_values)

    X, y = [], []
    for i in range(lookback, len(co2_scaled) - pred_horizon + 1):
        X.append(co2_scaled[i - lookback:i, 0])
        y.append(co2_scaled[i:i + pred_horizon, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))

    return X, y, scaler


if __name__ == "__main__":
    data_path = 'simulated_data/industry_co2_emissions.csv'
    industry_id = 5
    X, y, scaler = load_and_preprocess(data_path, industry_id)

    print(f"Prepared data for industry {industry_id}: X shape {X.shape}, y shape {y.shape}")

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Show a sample input sequence (scaled)
    print("Sample input sequence (scaled):")
    print(X[0].flatten())

    # Show corresponding output sequence (scaled)
    print("Sample output sequence (scaled):")
    print(y[0].flatten())

    # Inverse transform a sample input to real scale (CO2 kg)
    sample_real_input = scaler.inverse_transform(X[0].reshape(-1, 1)).flatten()
    print("Sample input sequence (real scale, kg CO2):")
    print(sample_real_input)
