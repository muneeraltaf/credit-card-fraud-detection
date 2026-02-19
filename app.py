import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This system predicts whether a transaction is **Fraudulent (1)** or **Genuine (0)**.

⚠ Features V1–V28 are anonymized PCA-transformed components.
""")

st.divider()

# -------------------------------
# MODEL PERFORMANCE SECTION
# -------------------------------

with st.expander("📊 Model Performance Summary"):

    st.markdown("""
    **Model Used:** Logistic Regression  
    **ROC-AUC Score:** 0.98  
    **Recall (Fraud Class):** 0.92  
    **Precision (Fraud Class):** ~0.90  
    """)

    st.markdown("""
    ### Why Recall is Important?

    In fraud detection systems, missing a fraudulent transaction 
    (False Negative) is more costly than incorrectly flagging a genuine transaction.

    Therefore, model selection prioritized **Recall for the Fraud class**.
    """)

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------

col1, col2 = st.columns(2)

input_data = []

with col1:
    st.subheader("Transaction Features (V1 - V15)")
    for i in range(15):
        value = st.number_input(f"V{i+1}", value=0.0)
        input_data.append(value)

with col2:
    st.subheader("Transaction Features (V16 - V30)")
    for i in range(15, 30):
        value = st.number_input(f"V{i+1}", value=0.0)
        input_data.append(value)

st.divider()

if st.button("Use Sample Transaction"):
    input_data = list(np.random.normal(0, 1, 30))
    st.success("Sample data generated. Click Predict.")

if st.button("Predict Transaction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")

    st.write(f"Fraud Probability: **{probability:.2%}**")

    st.progress(float(probability))

