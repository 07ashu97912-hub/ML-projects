import streamlit as st
import joblib
import numpy as np

# Title of the web app
st.title("Salary Prediction App 💰")
st.write("Welcome! This app predicts salary based on your experience.")

# 1. Model Selection Dropdown (Ready for your future models!)
selected_model = st.selectbox(
    "Choose a Machine Learning Model",
    ["Linear Regression (Current)", "Advanced Model (Coming Soon)"]
)

# 2. User Input Elements
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# 3. Prediction Logic
if st.button("Predict Salary"):
    if selected_model == "Linear Regression (Current)":
        try:
            # Load your saved model file
            lr = joblib.load("salary_model.pkl")
            ss = joblib.load("scaler.pkl")
            # Format input for scikit-learn (needs to be a 2D array)
            input_data = np.array([[experience]])
            scaled_input = ss.transform(input_data)
            prediction = float(lr.predict(scaled_input)[0])
            st.success(f"Estimated Salary: ${prediction:,.2f}")

            # Display result
            st.success(f"Estimated Salary: ${prediction:,.2f}")
        except FileNotFoundError:
            st.error("Error: 'salary_model.pkl' not found. Please ensure the model file is uploaded.")

    elif selected_model == "Advanced Model (Coming Soon)":
        st.info("This model is currently under development. Stay tuned!")
