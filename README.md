# 📊 Customer Churn Prediction App

This project is a **Machine Learning web application** built with **Streamlit** to predict whether a customer will churn (leave the bank) based on their profile.

---

## 🚀 Live Demo

👉 * Streamlit link here* https://churnappapp-fbdhuvdqgv36zsxfrnevb7.streamlit.app/

---

## 📌 Project Overview

Customer churn is a critical problem for businesses.
This project uses a **classification model (XGBoost)** to predict churn probability and help companies take proactive actions.

---

## 🧠 Model Details

* Algorithm: **XGBoost Classifier**
* Evaluation: Classification Report (Precision, Recall, F1-score)
* Threshold: **0.47** (custom decision boundary)

---

## 📊 Features Used

* Credit Score
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary
* Geography (One-Hot Encoded)
* Gender (One-Hot Encoded)

---

## ⚙️ Data Processing

* Removed unnecessary columns:

  * `id`, `CustomerId`, `Surname`
* Applied **One-Hot Encoding**:

  * Geography → France, Germany, Spain
  * Gender → Male, Female
* Ensured feature alignment between training and prediction

---

## 🖥️ App Features

* Interactive user input form
* Real-time churn prediction
* Probability output
* Clear result display (Churn / Not Churn)

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

## 📂 Project Structure

```
churn-app/
│── app.py
│── model.pkl
│── features.pkl
│── threshold.pkl
│── requirements.txt
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Sambat-Ms/churn_streamlit_app.git
cd churn_streamlit_app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed using **Streamlit Cloud**.

---

## 📈 Future Improvements

* Add model explainability (SHAP)
* Improve UI design
* Add more features for better accuracy
* Hyperparameter tuning

---

## 🙌 Author

**Sambat**



