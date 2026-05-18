import streamlit as st
import numpy as np
import pickle

# Load Model
with open("adaboost_classifier_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit Page Config
st.set_page_config(
    page_title="AdaBoost Disease Prediction",
    layout="centered"
)

# Title
st.title("🩺 Disease Prediction using AdaBoost Classifier")

st.write("Enter patient health details to predict disease risk.")

# User Inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=500,
    value=180
)

glucose = st.number_input(
    "Glucose",
    min_value=40,
    max_value=400,
    value=100
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=22.5
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=220,
    value=80
)

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

# Convert Smoking
smoking_value = 1 if smoking == "Yes" else 0

# Prediction Button
if st.button("Predict Disease"):

    input_data = np.array([[
        age,
        blood_pressure,
        cholesterol,
        glucose,
        bmi,
        heart_rate,
        smoking_value
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction Probability
    probability = model.predict_proba(input_data)[0]

    # Display Result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk of Disease Detected")
    else:
        st.success("Low Risk of Disease")

    st.write(f"Disease Probability: {probability[1]:.2f}")
    st.write(f"No Disease Probability: {probability[0]:.2f}")

st.markdown("---")
