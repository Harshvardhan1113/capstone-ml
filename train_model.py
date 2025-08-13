import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from data_prep import load_and_preprocess

def build_model(input_shape, pred_horizon):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(pred_horizon))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train(industry_id=5):
    data_path = 'simulated_data/industry_co2_emissions.csv'
    lookback = 24
    pred_horizon = 24
    X, y, scaler = load_and_preprocess(data_path, industry_id, lookback, pred_horizon)

    # Flatten y to 2D for model output shape
    y = y.reshape((y.shape[0], y.shape[1]))

    model = build_model((lookback, 1), pred_horizon)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    model.save(f'models/sim_industry_{industry_id}.h5')
    print(f"Model saved as models/sim_industry_{industry_id}.h5")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    train()
