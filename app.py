import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')
threshold = joblib.load('threshold.pkl')

st.title("Customer Churn Prediction")

# Inputs
CreditScore = st.number_input("Credit Score", 300, 900, 600)
Age = st.number_input("Age", 18, 100, 35)
Tenure = st.number_input("Tenure", 0, 10, 5)
Balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
NumOfProducts = st.number_input("Num Of Products", 1, 4, 1)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Categorical inputs
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    # ------------------------
    # CREATE RAW INPUT
    # ------------------------
    data = pd.DataFrame([{
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography': geography,
        'Gender': gender
    }])

    # ------------------------
    # ONE-HOT ENCODING (MATCH TRAINING)
    # ------------------------
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=False)

    # ------------------------
    # ALIGN FEATURES (FIXED)
    # ------------------------
    data = data.reindex(columns=features, fill_value=0)

    # ------------------------
    # PREDICT
    # ------------------------
    prob = model.predict_proba(data)[:, 1]
    pred = (prob > threshold).astype(int)

    st.write(f"Probability of churn: {prob[0]:.2f}")

    if pred[0] == 1:
        st.error("Customer WILL churn")
    else:
        st.success("Customer will NOT churn")