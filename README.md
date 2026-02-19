# 💳 Credit Card Fraud Detection System

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions using supervised learning techniques.

The dataset is highly imbalanced, and special techniques were used to handle class imbalance.

---

## 🚀 Technologies Used
- Python
- Scikit-learn
- Pandas & NumPy
- SMOTE (Imbalanced-Learn)
- Streamlit (Deployment)
- Joblib

---

## 📊 Problem Type
Binary Classification:
- 0 → Genuine Transaction
- 1 → Fraudulent Transaction

---

## ⚖ Handling Class Imbalance
Used SMOTE (Synthetic Minority Oversampling Technique) to balance training data.

---

## 🤖 Models Implemented
- Logistic Regression
- Random Forest

Model selection was based on Recall for Class 1 (Fraud), as detecting fraudulent transactions is more critical than overall accuracy.

---

## 📈 Final Results

| Model | ROC-AUC | Recall (Fraud) |
|--------|----------|----------------|
| Logistic Regression | 0.98 | 0.92 |
| Random Forest | 0.98 | 0.85 |

Final model selected: **Logistic Regression**

---

## 🌐 Deployment

The model is deployed locally using Streamlit.

To run the app:

```bash
streamlit run app.py
