import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('ScalerLA.pkl')


# App UI
st.title("🔍 Loan Approval Predictor")
st.markdown("All features (including categorical) will be scaled like in training")

with st.form("loan_form"):
    # Input fields
    st.subheader("Applicant Details")
    married = st.radio("Married", ["No", "Yes"], index=1)
    education = st.radio("Education", ["Not Graduate", "Graduate"], index=1)
    income = st.number_input("Monthly Income (₹)", min_value=0, value=3000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
    credit_history = st.radio("Credit History", ["Unclear", "Clear"], index=1)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input array (match training column order!)
    input_data = np.array([[
        1 if married == "Yes" else 0,       # Married (1/0)
        1 if education == "Graduate" else 0, # Education (1/0)
        income,                              # ApplicantIncome
        loan_amount,                         # LoanAmount
        1 if credit_history == "Clear" else 0 # Credit_History (1/0)
    ]])
    
    # Apply scaling to ALL features (critical!)
    scaled_input = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]  # Confidence
    
    # Display result
    st.subheader("Result")
    if prediction == 1:
        st.success("✅ Approved")
    else:
        st.error("❌ Denied")
    
    # Debug: Show scaled values (optional)
    st.caption("Scaled Features Used for Prediction:")
    st.write(scaled_input)