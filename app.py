import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="PayGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #07152f 0%, #0b1f45 55%, #08152c 100%); color: #f4f8ff;}
[data-testid="stMetric"] {background: rgba(20,54,105,.55); border: 1px solid #1e5ca4; padding: 16px; border-radius: 12px;}
div[data-testid="stForm"] {background: rgba(15,43,84,.55); padding: 18px; border-radius: 14px; border: 1px solid #24518a;}
h1, h2, h3 {color: #f4f8ff;}
.small-muted {color: #a9c4e8; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ PayGuard AI")
st.caption("Smarter Detection. Safer Payments. Greater Trust.")
st.info("Prototype notice: This demo analyzes synthetic or user-provided transaction data. It is not connected to PayPal, does not process payments, and must not be used to make real financial decisions.")

REQUIRED = ["amount", "country", "is_new_device", "transactions_last_hour",
            "account_age_days", "distance_from_last_km", "is_international"]

def make_reference_data(n=1200, seed=17):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": np.clip(rng.lognormal(3.2, .8, n), 1, 2500),
        "transactions_last_hour": rng.poisson(1.1, n),
        "account_age_days": rng.integers(30, 2500, n),
        "distance_from_last_km": np.clip(rng.exponential(18, n), 0, 500),
        "is_new_device": rng.binomial(1, .12, n),
        "is_international": rng.binomial(1, .10, n),
    })

@st.cache_resource
def train_anomaly_model():
    ref = make_reference_data()
    model = IsolationForest(n_estimators=160, contamination=0.04, random_state=42)
    model.fit(ref[["amount", "transactions_last_hour", "account_age_days",
                   "distance_from_last_km", "is_new_device", "is_international"]])
    return model

model = train_anomaly_model()

def analyze(row):
    reasons, score = [], 0
    amount = float(row["amount"])
    tx = int(row["transactions_last_hour"])
    age = int(row["account_age_days"])
    distance = float(row["distance_from_last_km"])
    new_device = int(row["is_new_device"])
    international = int(row["is_international"])

    if amount >= 1000:
        score += 22; reasons.append("Unusually high transaction amount (≥ 1,000).")
    elif amount >= 500:
        score += 12; reasons.append("Higher-than-usual transaction amount (≥ 500).")
    if tx >= 6:
        score += 25; reasons.append("High transaction velocity: 6 or more transactions in one hour.")
    elif tx >= 3:
        score += 12; reasons.append("Elevated transaction velocity: 3–5 transactions in one hour.")
    if age < 7:
        score += 18; reasons.append("Account is very new (less than 7 days old).")
    elif age < 30:
        score += 9; reasons.append("Account is relatively new (less than 30 days old).")
    if distance >= 500:
        score += 20; reasons.append("Large location change compared with the previous transaction (≥ 500 km).")
    elif distance >= 200:
        score += 12; reasons.append("Unusual location change compared with the previous transaction (≥ 200 km).")
    if new_device:
        score += 12; reasons.append("Transaction is coming from a new device.")
    if international:
        score += 8; reasons.append("Transaction is marked as international.")

    features = pd.DataFrame([{
        "amount": amount, "transactions_last_hour": tx, "account_age_days": age,
        "distance_from_last_km": distance, "is_new_device": new_device,
        "is_international": international
    }])
    anomaly = model.decision_function(features)[0]
    anomaly_flag = model.predict(features)[0] == -1
    if anomaly_flag:
        score += 10
        reasons.append("Machine-learning anomaly model identified an unusual combination of signals.")

    score = min(score, 100)
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    if not reasons:
        reasons.append("No major risk indicators were triggered by the current demo checks.")
    return score, level, reasons, bool(anomaly_flag)

def normalize_data(df):
    df = df.copy()
    for col in ["amount", "transactions_last_hour", "account_age_days", "distance_from_last_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["is_new_device", "is_international"]:
        df[col] = df[col].replace({"yes": 1, "no": 0, "true": 1, "false": 0, True: 1, False: 0})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["country"] = df["country"].astype(str)
    return df.dropna(subset=REQUIRED)

tab1, tab2, tab3 = st.tabs(["🔎 Analyze a transaction", "📊 Transaction explorer", "ℹ️ About & data guide"])

with tab1:
    st.subheader("Check a payment transaction")
    st.write("Enter example transaction details to receive a transparent risk assessment.")
    with st.form("transaction_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction amount (demo units)", min_value=0.0, value=125.0, step=10.0)
            country = st.text_input("Transaction country", value="Nepal")
            new_device = st.selectbox("New device?", ["No", "Yes"])
            international = st.selectbox("International transaction?", ["No", "Yes"])
        with c2:
            tx_count = st.number_input("Transactions in the last hour", min_value=0, max_value=100, value=1)
            account_age = st.number_input("Account age (days)", min_value=0, max_value=10000, value=365)
            distance = st.number_input("Distance from previous transaction (km)", min_value=0.0, value=5.0, step=5.0)
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)

    if submitted:
        row = {"amount": amount, "country": country, "is_new_device": int(new_device == "Yes"),
               "transactions_last_hour": tx_count, "account_age_days": account_age,
               "distance_from_last_km": distance, "is_international": int(international == "Yes")}
        score, level, reasons, anomaly = analyze(row)
        st.divider()
        st.subheader("Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score", f"{score}/100")
        m2.metric("Risk level", level)
        m3.metric("ML anomaly flag", "Unusual" if anomaly else "Not flagged")
        st.progress(score / 100)
        if level == "High":
            st.error("High-risk indicators detected. Review this transaction carefully.")
        elif level == "Medium":
            st.warning("Some risk indicators detected. Additional verification may be appropriate.")
        else:
            st.success("Low risk under the current demo checks. This does not guarantee the transaction is safe.")
        st.markdown("#### Why was it flagged?")
        for reason in reasons:
            st.write("• " + reason)
        st.caption("Risk score is an experimental demonstration score, not a calibrated probability of fraud.")

with tab2:
    st.subheader("Explore transaction data")
    st.write("Use the included sample file or upload a CSV with the required columns.")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_csv(Path(__file__).parent / "sample_transactions.csv")
        st.caption("Showing the included synthetic sample dataset.")
    clean = normalize_data(raw)
    if clean.empty:
        st.error("No valid rows found. Check the required columns in the data guide.")
    else:
        results = []
        for _, r in clean.iterrows():
            score, level, reasons, anomaly = analyze(r)
            results.append({"risk_score": score, "risk_level": level,
                            "explanation": " ".join(reasons), "ml_anomaly": anomaly})
        view = pd.concat([clean.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        a, b, c = st.columns(3)
        a.metric("Transactions analyzed", len(view))
        b.metric("High-risk transactions", int((view.risk_level == "High").sum()))
        c.metric("Average demo risk score", f"{view.risk_score.mean():.1f}/100")
        counts = view["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk level").reset_index(name="Count")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(counts, x="Risk level", y="Count", color="Risk level",
                                   title="Transactions by risk level"), use_container_width=True)
        with right:
            st.plotly_chart(px.histogram(view, x="risk_score", nbins=15,
                                         title="Distribution of demo risk scores"), use_container_width=True)
        st.markdown("#### Analyzed transactions")
        st.dataframe(view, use_container_width=True, hide_index=True)
        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download analysis results", csv, "payguard_analysis.csv", "text/csv")

with tab3:
    st.subheader("About PayGuard AI")
    st.write("PayGuard AI is a hackathon prototype exploring how transaction signals and anomaly detection can support payment security.")
    st.markdown("""
**Current prototype capabilities**
- Transparent rule-based risk indicators.
- An Isolation Forest anomaly detector trained on generated reference data.
- Single-transaction analysis.
- CSV batch analysis, charts, and downloadable results.

**CSV format**

Your CSV should include these columns:

| Column | Example | Meaning |
|---|---:|---|
| `amount` | 125.50 | Transaction amount in a consistent unit |
| `country` | Nepal | Transaction country |
| `is_new_device` | 0 or 1 | Whether the device is new |
| `transactions_last_hour` | 2 | Number of recent transactions |
| `account_age_days` | 365 | Account age in days |
| `distance_from_last_km` | 5 | Distance from previous transaction |
| `is_international` | 0 or 1 | Whether the transaction is international |

**Important limitations**
- The included data is synthetic and is not PayPal customer data.
- The model is a demonstration anomaly detector, not a production fraud model.
- A high risk score does not prove fraud, and a low score does not prove safety.
- Do not upload real personal, payment-card, or sensitive financial information.
- This prototype does not connect to PayPal or authorize, block, or process payments.
""")

st.divider()
st.caption("PayGuard AI • Hackathon prototype • Smarter Detection. Safer Payments. Greater Trust.")
