import os
import joblib
import numpy as np
import streamlit as st

# Title of the web app
st.title("Salary Prediction App 💰")
st.write("Welcome! This app predicts salary based on your experience.")

# 1. Model Selection Dropdown
selected_model = st.selectbox(
    "Choose a Machine Learning Model",
    ["Linear Regression (Current)", "Advanced Model (Coming Soon)"],
)

# Fix pathing for Streamlit Community Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "salary_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# Cache the loading process so your app stays fast and doesn't reload files on every click
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Missing file: 'salary_model.pkl' was not found at {MODEL_PATH}"
        )
        return None, None
    if not os.path.exists(SCALER_PATH):
        st.error(f"Missing file: 'scaler.pkl' was not found at {SCALER_PATH}")
        return None, None

    lr = joblib.load(MODEL_PATH)
    ss = joblib.load(SCALER_PATH)
    return lr, ss


# Load the model and scaler
lr, ss = load_assets()

# 2. User Input Elements
experience = st.number_input(
    "Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5
)

# 3. Prediction Logic
if st.button("Predict Salary"):
    if selected_model == "Linear Regression (Current)":
        # Make sure assets loaded successfully before predicting
        if lr is not None and ss is not None:
            try:
                # Format input for scikit-learn (needs to be a 2D array)
                input_data = np.array([[experience]])
                scaled_input = ss.transform(input_data)
                prediction = float(lr.predict(scaled_input)[0])

                # Display result (Removed the duplicate success box)
                st.success(f"Estimated Salary: ${prediction:,.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error(
                "Cannot predict. Please fix the missing files listed above."
            )

    elif selected_model == "Advanced Model (Coming Soon)":
        st.info("This model is currently under development. Stay tuned!")
