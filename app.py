import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bank Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD FILES
# =========================
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')
threshold = joblib.load('threshold.pkl')

# =========================
# HEADER
# =========================
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict whether a customer will leave the bank based on key attributes.")
st.divider()

# =========================
# SIDEBAR (OPTIONAL CLEAN INFO)
# =========================
st.sidebar.title("ℹ About App")
st.sidebar.info("This app uses Machine Learning to predict customer churn probability.")

# =========================
# INPUT SECTION
# =========================
st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    CreditScore = st.number_input("Credit Score", 300, 900, 600)
    Age = st.number_input("Age", 18, 100, 35)
    Tenure = st.number_input("Tenure", 0, 10, 5)

with col2:
    Balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    NumOfProducts = st.number_input("Num Of Products", 1, 4, 1)
    EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

with col3:
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Churn"):

    # ------------------------
    # CREATE INPUT DATA
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
    # ONE HOT ENCODING
    # ------------------------
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=False)

    # ------------------------
    # ALIGN FEATURES
    # ------------------------
    data = data.reindex(columns=features, fill_value=0)

    # ------------------------
    # PREDICTION
    # ------------------------
    prob = float(model.predict_proba(data)[:, 1][0])
    pred = int(prob > threshold)

    # =========================
    # RESULT UI (IMPROVED)
    # =========================
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        st.metric("Threshold", f"{threshold:.2f}")

    st.divider()

    if pred == 1:
        st.error("⚠ Customer is HIGH RISK of Churn")
        st.warning("Recommended action: Offer retention strategy")
    else:
        st.success("✔ Customer is LOW RISK of Churn")
        st.info("Customer is likely to stay")

    # =========================
    # PROBABILITY BAR (OPTIONAL NICE TOUCH)
    # =========================
    st.progress(min(prob, 1.0))