import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

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
# SIDEBAR
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

    # =========================
    # CREATE INPUT DATA
    # =========================
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

    # =========================
    # ENCODE FEATURES
    # =========================
    data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=False)
    data_encoded = data_encoded.reindex(columns=features, fill_value=0)

    # =========================
    # PREDICTION
    # =========================
    prob = float(model.predict_proba(data_encoded)[:, 1][0])
    pred = int(prob > threshold)

    # =========================
    # RESULT UI
    # =========================
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        st.metric("Threshold", f"{threshold:.2f}")

    st.progress(min(prob, 1.0))

    st.divider()

    # =========================
    # RISK DISPLAY (COLOR FIXED)
    # =========================
    if pred == 1:
        st.markdown(
            "<h3 style='color:red;'>⚠ HIGH RISK: Customer will CHURN</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color:green;'>✔ LOW RISK: Customer will STAY</h3>",
            unsafe_allow_html=True
        )

    # =========================
    # PIE CHART
    # =========================
    st.subheader("📈 Churn Risk Distribution")

    pie_fig = go.Figure(data=[go.Pie(
        labels=["Churn Risk", "Retention Chance"],
        values=[prob, 1 - prob],
        hole=0.4,
        marker=dict(colors=["#ff4b4b", "#00c853"])
    )])

    st.plotly_chart(pie_fig, use_container_width=True)

    # =========================
    # 👤 CUSTOMER PROFILE SUMMARY
    # =========================
    st.subheader("👤 Customer Profile Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.write("### 💰 Financial Info")
        st.write(f"💰 Balance: {Balance}")
        st.write(f"💳 Credit Score: {CreditScore}")
        st.write(f"💵 Estimated Salary: {EstimatedSalary}")

    with summary_col2:
        st.write("### 🏦 Account Info")
        st.write(f"📦 Products: {NumOfProducts}")
        st.write(f"📅 Tenure: {Tenure} years")
        st.write(f"🏦 Has Credit Card: {'Yes' if HasCrCard == 1 else 'No'}")

    with summary_col3:
        st.write("### 👤 Behavior Info")
        st.write(f"👤 Age: {Age}")
        st.write(f"🌍 Geography: {geography}")
        st.write(f"⚡ Active Member: {'Yes' if IsActiveMember == 1 else 'No'}")