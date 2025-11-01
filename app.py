import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd

# Load model and encoders
model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Run prediction when button clicked
if st.button('Predict'):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

    # Encode categorical variables
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    geo_encoded = pd.DataFrame(
        onehot_encoder_geo.transform(input_data[['Geography']]),
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine and scale
    input_final = pd.concat([input_data.drop(columns=['Geography']), geo_encoded], axis=1)
    input_scaled = scaler.transform(input_final)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])

    # Display results
    st.write("Probability of churn:", prediction_proba)

    if prediction_proba > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to not churn.")
