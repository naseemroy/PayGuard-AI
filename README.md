# PayGuard AI — MVP

A Streamlit hackathon prototype for exploring payment transaction risk signals.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal (usually http://localhost:8501).

## Included features

- Single transaction analysis form.
- Transparent rule-based risk scoring.
- Scikit-learn Isolation Forest anomaly flag trained on generated reference data.
- CSV batch analysis, charts, and downloadable results.
- Synthetic sample transaction dataset.

## CSV columns

`amount`, `country`, `is_new_device`, `transactions_last_hour`, `account_age_days`, `distance_from_last_km`, `is_international`

Boolean fields accept 0/1 or yes/no.

## Important

This is an experimental demonstration, not a production fraud detection system. The risk score is not a calibrated probability. The reference model is trained on synthetic data. Do not upload real customer, card, or sensitive financial data. This app is not connected to PayPal and does not process or block payments.
