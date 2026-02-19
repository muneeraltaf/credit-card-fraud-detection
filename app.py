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

⚠ Note: Features V1–V28 are anonymized PCA-transformed components.
Users can input sample values for demonstration purposes.
""")

st.divider()

# Create two columns layout
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

# Sample Data Button
if st.button("Use Sample Transaction"):
    input_data = list(np.random.normal(0, 1, 30))
    st.success("Sample data generated. Click Predict.")

if st.button("Predict Transaction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")
