import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load sample transactions
sample_df = pd.read_csv("sample_transactions.csv")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This system predicts whether a transaction is **Fraudulent (1)** or **Genuine (0)**.

Instead of manually entering PCA features, you can now select real sample transactions.
""")

st.divider()

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

with st.expander("📊 Model Performance Summary"):
    st.markdown("""
    **Model Used:** Logistic Regression  
    **ROC-AUC Score:** 0.98  
    **Recall (Fraud Class):** 0.92  
    """)

st.divider()

# -------------------------------
# SAMPLE SELECTION
# -------------------------------

st.subheader("🔎 Select a Sample Transaction")

transaction_type = st.radio(
    "Choose Transaction Type:",
    ("Genuine Transaction", "Fraudulent Transaction")
)

if transaction_type == "Genuine Transaction":
    selected_row = sample_df[sample_df["Class"] == 0].sample(1)
else:
    selected_row = sample_df[sample_df["Class"] == 1].sample(1)

st.write("Selected Transaction Preview:")
st.dataframe(selected_row)

st.divider()

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Selected Transaction"):

    X_input = selected_row.drop("Class", axis=1).values
    X_scaled = scaler.transform(X_input)

    probability = model.predict_proba(X_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("📢 Prediction Result")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")

    st.write(f"Fraud Probability: **{probability:.2%}**")
    st.progress(float(probability))
