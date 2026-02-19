# 💳 Credit Card Fraud Detection System

## 🌍 Live Demo
🔗 https://credit-card-fraud-detection-2ji7xzzirr6uwqpuj97rie.streamlit.app/ 

---

## 📌 Project Overview

This project builds and deploys an end-to-end Machine Learning system to detect fraudulent credit card transactions.

The dataset is highly imbalanced, and special care was taken to handle class imbalance and select the model based on business-critical evaluation metrics.

The final solution is deployed as an interactive web application using Streamlit.

---

## 🚀 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- SMOTE (Imbalanced-Learn)
- Logistic Regression
- Random Forest (comparison)
- Streamlit (Web Deployment)
- Joblib
- Git & GitHub
- Streamlit Community Cloud

---

## 📊 Problem Type

Binary Classification:

- 0 → Genuine Transaction  
- 1 → Fraudulent Transaction  

---

## ⚖ Handling Class Imbalance

The dataset contains significantly fewer fraudulent transactions compared to genuine ones.

To address this imbalance:

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data
- Evaluated models using **Recall for Fraud class**
- Avoided relying on accuracy alone

---

## 🤖 Model Development

### Models Trained
- Logistic Regression
- Random Forest

### Model Selection Criteria

Instead of selecting the model purely based on ROC-AUC, model selection prioritized:

- Recall for Fraud class
- Business impact (minimizing false negatives)

### Final Model Selected
**Logistic Regression**

Reason:
It achieved higher recall for fraudulent transactions compared to Random Forest.

---

## 📈 Model Performance

| Metric | Logistic Regression |
|--------|---------------------|
| ROC-AUC | ~0.98 |
| Recall (Fraud) | 0.92 |
| Precision (Fraud) | ~0.90 |

### Why Recall Was Prioritized?

In fraud detection systems, missing a fraudulent transaction (False Negative) is more costly than incorrectly flagging a genuine one.

Therefore, the model was selected based on its ability to detect fraudulent transactions effectively.

---

## 🧠 Feature Explanation

The dataset is anonymized for confidentiality.

Original transaction features were transformed using **Principal Component Analysis (PCA)**, resulting in features:

- V1 to V28 (PCA components)
- Time
- Amount

These features are not human-interpretable and are automatically generated from transaction metadata in real-world systems.

---

## 🌐 Application Features

The deployed Streamlit application includes:

- 📊 Model Performance Summary section
- 🔎 Realistic sample transaction selection (Fraud / Genuine)
- 📈 Fraud probability display
- 📉 Confidence progress bar
- 🎯 Business-focused prediction explanation

Instead of manually entering PCA values, users can simulate realistic transactions using real dataset samples.

---

## 📂 Project Structure

credit_card_fraud_project/
│
├── notebooks/
│ └── EDA.ipynb
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── sample_transactions.csv
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run Locally

1. Clone the repository

git clone https://github.com/muneeraltaf/credit-card-fraud-detection.git


2. Navigate into project folder

cd credit-card-fraud-detection


3. Create virtual environment

python3 -m venv venv
source venv/bin/activate


4. Install dependencies

pip install -r requirements.txt


5. Run the app

streamlit run app.py


---

## 📥 Dataset

Dataset not included due to GitHub file size limitations.

Download from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🎯 Key Learnings

- Handling severely imbalanced datasets
- Proper metric selection in real-world ML problems
- Feature scaling and model convergence
- Model comparison and selection strategy
- Deployment of ML models using Streamlit
- Version control using Git & GitHub
- Cloud deployment workflow

---

## 👨‍💻 Muneer Altaf

Built as part of practical Machine Learning training and deployment practice.