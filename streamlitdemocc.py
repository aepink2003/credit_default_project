import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Default Risk Predictor", layout="centered")

st.title("Credit Card Default Risk Predictor")
st.write("Predict default risk based on customer credit behavior.")

# Load trained model
@st.cache_resource
def load_model():
    return pickle.load(open("xgb_model.pkl", "rb"))

model = load_model()

st.subheader("Customer Information")

limit_bal = st.number_input("Credit Limit (NT$)", min_value=0, value=50000)
age = st.number_input("Age", min_value=18, max_value=100, value=35)

st.subheader("Recent Payment Behavior")

max_delay = st.selectbox(
    "Maximum Payment Delay (months)",
    options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="-1 = paid on time, 1 = 1 month late, etc."
)

avg_delay = st.number_input(
    "Average Payment Delay (months)",
    min_value=-1.0,
    max_value=9.0,
    value=0.5,
    help="Average of repayment delay codes over recent months"
)

st.subheader("Billing & Payments")

avg_bill = st.number_input("Average Monthly Bill (NT$)", min_value=0.0, value=20000.0)
avg_payment = st.number_input("Average Monthly Payment (NT$)", min_value=0.0, value=15000.0)

# Create input dataframe
input_df = pd.DataFrame([{
    "LIMIT_BAL": limit_bal,
    "AGE": age,
    "max_delay": max_delay,
    "avg_delay": avg_delay,
    "avg_bill": avg_bill,
    "avg_payment": avg_payment
}])

# Prediction
if st.button("Predict Default Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Default Risk Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("High Risk: Recommend intervention")
    else:
        st.success("Low Risk")
