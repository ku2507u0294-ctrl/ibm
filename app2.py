import streamlit as st
import pandas as pd
import pickle
import os

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# Custom CSS for a slightly different look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction System")
st.write("Enter the customer details below to predict if they are likely to churn.")

# Load the model
try:
    with open('model2.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please run train2.py first.")
    st.stop()

# Input form
with st.form("prediction_form"):
    st.subheader("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.5)
        
    with col2:
        sub_length = st.number_input("Subscription Length (Months)", min_value=0, max_value=120, value=12)
        support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=1)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Monthly_Charges': [monthly_charges],
        'Subscription_Length_Months': [sub_length],
        'Support_Calls': [support_calls]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.1%})")
        st.write("Recommendation: Offer a discount or contact them immediately to resolve issues.")
    else:
        st.success(f"✅ Low Risk of Churn. (Probability: {probability:.1%})")
        st.write("Recommendation: Customer is likely satisfied. Continue regular engagement.")
