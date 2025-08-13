# Carbon Emission Monitoring & Compliance Forecasting

An AI-powered pipeline to monitor, analyze, and forecast industrial carbon emissions.  
Integrates IoT-based real-time data collection, ARIMA-based forecasting, and compliance checks to flag industries exceeding their emission quotas.

---

## Overview
This project is part of a sustainability initiative to ensure industries remain within their assigned carbon emission limits.  
The system:
- Ingests real-time or historical emission data.
- Trains an ARIMA model to forecast emissions for the next month.
- Checks forecasts against compliance quotas.
- Generates alerts/reports for any potential violations.
- Can integrate with blockchain for transparent record-keeping.

---

## Features
- **Real-time monitoring** of carbon emissions from IoT sensors.
- **Forecasting** next month’s emissions using ARIMA.
- **Compliance checks** against assigned quotas.
- **Flagging mechanism** for industries likely to exceed limits.
- **Automated report generation** for stakeholders.

---

## Tech Stack
- **Python** (Data processing, forecasting, compliance logic)
- **Pandas** (Data handling)
- **Statsmodels** (ARIMA forecasting)
- **Matplotlib** (Optional visualization)
- **HDF5 (.h5)** for storing trained models
- *(Optional)* Blockchain layer for immutable compliance logs

---

##  How It Works
1. **Data Input**  
   - CSV or database with `timestamp` and `emission` fields.
   - Example:  
     ```
     2025-06-01 00:00:00, 142.5
     2025-06-02 00:00:00, 139.8
     ...
     ```

2. **Model Training**  
   - ARIMA model is trained on historical emission data.
   - Model is saved as `.h5` for reuse.

3. **Forecasting & Compliance**  
   - Predicts next month’s total emissions.
   - Compares predictions with compliance quota.
   - Flags industries likely to exceed limits.

4. **Reporting**  
   - Generates text-based or JSON reports for compliance officers.

---


---

## Installation
```bash
git clone https://github.com/Harshvardhan1113/capstone-ml.git

