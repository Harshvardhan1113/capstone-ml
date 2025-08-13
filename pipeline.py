import pandas as pd
import numpy as np
import tensorflow as tf
from data_prep import load_and_preprocess
from compliance import compliance_report
from datetime import datetime
import os

def predict_and_evaluate(industry_id=5):
    data_path = 'simulated_data/industry_co2_emissions.csv'
    lookback = 24
    pred_horizon = 24

    X, y, scaler = load_and_preprocess(data_path, industry_id, lookback, pred_horizon)

    model_path = f'models/sim_industry_{industry_id}.h5'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please train first.")
        return

    model = tf.keras.models.load_model(model_path)

    preds_scaled = model.predict(X)
    # Inverse scale predictions
    preds = scaler.inverse_transform(preds_scaled)

    # For compliance, aggregate last 30 days monthly emission estimate
    df = pd.read_csv(data_path)
    df = df[df['industry_id'] == industry_id].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Sum last 30 days hourly co2 in kg
    last_30_days = df['co2_kg'].tail(30 * 24).sum()

    report = compliance_report(last_30_days)

    print(f"Industry {industry_id} Compliance Report:")
    print(f"Monthly CO2 Emission (kg): {report['monthly_emission_kg']:.2f}")
    print(f"Credits Allocated: {report['credits_allocated']:.2f}")
    print(f"Status: {'Approved' if report['approved'] else 'Rejected'}")
    print(f"Reason: {report['reason']}")

if __name__ == "__main__":
    predict_and_evaluate()
