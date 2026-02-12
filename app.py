import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Probability", page_icon="📉", layout="centered")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "churn_demo_model.joblib"
COLS_PATH = BASE / "models" / "churn_demo_columns.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_cols():
    with open(COLS_PATH, "r") as f:
        return json.load(f)

st.title("Customer Churn Probability")
st.write("This demo predicts churn using 5 simple customer details.")

model = load_model()
cols = load_cols()

# --- Inputs (only 5) ---
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

if st.button("Predict"):
    row = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
    }

    X = pd.DataFrame([row])[cols]

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    st.write(f"Churn probability: **{prob * 100:.1f}%**")
    if pred == 1:
        st.error("Likely to churn")
    else:
        st.success("Likely to stay")

st.caption(f"Loaded: {MODEL_PATH.name}")
