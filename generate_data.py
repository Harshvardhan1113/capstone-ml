import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def simulate_co2_emissions(num_hours, base_level=100, trend=0.001, seasonal_amplitude=50, noise_std=15, spike_prob=0.01):
    emissions = []
    for t in range(num_hours):
        trend_comp = trend * t
        daily_season = seasonal_amplitude * np.sin(2 * np.pi * (t % 24) / 24)
        day_of_week = (t // 24) % 7
        weekly_factor = 0.7 if day_of_week in [5, 6] else 1.0
        val = base_level + trend_comp + daily_season
        val += np.random.normal(0, noise_std)
        if np.random.rand() < spike_prob:
            val += np.random.uniform(50, 150)
        val *= weekly_factor
        val = max(val, 10)
        emissions.append(val)
    return emissions

def generate_dataset():
    num_industries = 5
    hours_per_day = 24
    days = 180
    total_hours = hours_per_day * days
    start_date = datetime(2025, 1, 1)
    time_index = [start_date + timedelta(hours=i) for i in range(total_hours)]

    data = []
    for industry_id in range(1, num_industries + 1):
        base_level = 80 + 40 * np.random.rand()
        trend = 0.0005 + 0.0015 * np.random.rand()
        seasonal_amplitude = 40 + 20 * np.random.rand()
        co2_emissions = simulate_co2_emissions(total_hours, base_level, trend, seasonal_amplitude)
        for i, dt in enumerate(time_index):
            data.append({
                'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'industry_id': industry_id,
                'co2_kg': co2_emissions[i],
                'interval_hours': 1
            })

    df = pd.DataFrame(data)
    os.makedirs('simulated_data', exist_ok=True)
    df.to_csv('simulated_data/industry_co2_emissions.csv', index=False)
    print("Dataset generated: simulated_data/industry_co2_emissions.csv")

if __name__ == "__main__":
    generate_dataset()
