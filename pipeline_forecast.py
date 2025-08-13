import pandas as pd
import numpy as np
import tensorflow as tf
from data_prep import load_and_preprocess
from compliance import compliance_report
from datetime import datetime, timedelta
import os

def _compute_next_month_window(last_ts: pd.Timestamp):
    """Return start (inclusive) and end (inclusive) timestamps for the next calendar month."""
    last_ts = pd.to_datetime(last_ts)
    next_month_start = (last_ts + pd.offsets.MonthBegin(1)).normalize()
    next_month_end = (next_month_start + pd.offsets.MonthEnd(0)) + pd.Timedelta(hours=23)
    return next_month_start, next_month_end

def _hours_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """Number of hourly steps between two timestamps inclusive."""
    return int(((end_ts - start_ts).total_seconds() // 3600) + 1)

def predict_and_evaluate(industry_id=5):
    data_path = 'simulated_data/industry_co2_emissions.csv'
    lookback = 24
    pred_horizon = 24

    # Load scaled windows for this industry (unchanged) 
    X, y, scaler = load_and_preprocess(data_path, industry_id, lookback, pred_horizon)

    model_path = f'models/sim_industry_{industry_id}.h5'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please train first.")
        return

    model = tf.keras.models.load_model(model_path)

    #  Original prediction (kept) + safe inverse scaling 
    preds_scaled = model.predict(X)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)

    #  Historical compliance over the last 30 days 
    df = pd.read_csv(data_path)
    df = df[df['industry_id'] == industry_id].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    last_30_days = df['co2_kg'].tail(30 * 24).sum()
    report_current = compliance_report(last_30_days)

    print(f"Industry {industry_id} Compliance Report (Actuals):")
    print(f"Monthly CO2 Emission (kg): {report_current['monthly_emission_kg']:.2f}")
    print(f"Credits Allocated: {report_current['credits_allocated']:.2f}")
    print(f"Status: {'Approved' if report_current['approved'] else 'Rejected'}")
    print(f"Reason: {report_current['reason']}")

 

    # Determine next month range
    last_ts = df['timestamp'].iloc[-1]
    next_month_start, next_month_end = _compute_next_month_window(last_ts)


    first_pred_ts = (last_ts + pd.Timedelta(hours=1)).floor('H')
    total_hours_needed = _hours_between(first_pred_ts, next_month_end)

    history_scaled = X[-1, :, 0].copy()  
    predicted_scaled_vals = []

    hours_generated = 0
    while hours_generated < total_hours_needed:
        # Predict next pred_horizon hours 
        block = model.predict(history_scaled.reshape(1, lookback, 1), verbose=0)[0]  
        remaining = total_hours_needed - hours_generated
        take = min(pred_horizon, remaining)

     
        predicted_scaled_vals.extend(block[:take].tolist())
        hours_generated += take

        # Update history window with the whole block 
    
        new_hist = np.concatenate([history_scaled, block])[-lookback:]
        history_scaled = new_hist

    # Inverse transform the scaled predictions back to kg
    predicted_vals = scaler.inverse_transform(np.array(predicted_scaled_vals).reshape(-1, 1)).flatten()

    # Build timestamps for each hourly prediction
    pred_timestamps = pd.date_range(first_pred_ts, periods=total_hours_needed, freq='H')

    # Keep only the hours that fall in the next calendar month
    mask_next_month = (pred_timestamps >= next_month_start) & (pred_timestamps <= next_month_end)
    next_month_pred_total = float(predicted_vals[mask_next_month].sum())

    # Run compliance on the forecasted next-month total
    report_forecast = compliance_report(next_month_pred_total)

    print("\nIndustry {} Forecast Compliance (Next Month):".format(industry_id))
    print(f"Forecast Month: {next_month_start.strftime('%Y-%m')} "
          f"({next_month_start.date()} to {next_month_end.date()})")
    print(f"Predicted CO2 Emission (kg): {report_forecast['monthly_emission_kg']:.2f}")
    print(f"Credits Allocated (forecast): {report_forecast['credits_allocated']:.2f}")
    print(f"Status (forecast): {'Approved' if report_forecast['approved'] else 'Rejected'}")
    print(f"Reason: {report_forecast['reason']}")

if __name__ == "__main__":
    predict_and_evaluate()
