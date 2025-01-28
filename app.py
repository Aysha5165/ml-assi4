import streamlit as st
import joblib

# Load the pre-trained SVM model
@st.cache_resource
def load_model():
    return joblib.load('scaler.pkl')  # Replace 'svm_model.pkl' with your actual model file name

model = load_model()

# Define the prediction function
def predict(input_data):
    prediction = model.predict([input_data])
    return prediction[0]  # Assuming prediction returns an array

# Streamlit GUI
st.title("SVM Prediction App")
st.write("This app predicts results using a pre-trained SVM model. Provide the required inputs below:")

# Input fields
st.subheader("Input Features")
input_1 = st.number_input("Feature 1 (e.g., numeric input):", step=0.01)
input_2 = st.number_input("Feature 2 (e.g., numeric input):", step=0.01)

# Collect user inputs
user_input = [input_1, input_2]

# Predict button
if st.button("Predict"):
    try:
        # Make prediction
        result = predict(user_input)
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

