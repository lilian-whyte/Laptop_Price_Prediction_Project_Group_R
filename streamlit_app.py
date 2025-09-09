#!/usr/bin/env python
import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- Set up the Streamlit page and styling ---
st.set_page_config(layout="wide")
st.title("Laptop Price Prediction")
st.markdown("""
<style>
body {
    background-color: #4B0082; /* Indigo background */
    color: #F0F2F6; /* Light text color */
    font-family: sans-serif;
}
.stApp {
    background-color: #4B0082; /* Ensure the main app container also has indigo background */
    color: #F0F2F6; /* Light text color */
}
.st-cy, .st-d1, .st-ch { /* Unify styling for input labels, number input, and selectbox */
    color: #E6E6FA; /* Lighter indigo for labels and text */
}
.st-d1, .st-ch {
    background-color: #6A5ACD; /* Slate blue background for input fields */
    border-radius: 5px;
    padding: 10px;
}
.stButton > button {
    background-color: #8A2BE2; /* Blueviolet button background */
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background-color: #9370DB; /* Mediumpurple on hover */
}
h1, h2, h3, h4, h5, h6 {
    color: #E6E6FA; /* Lighter indigo for headers */
}
</style>
""", unsafe_allow_html=True)

# --- Define the paths for the model and preprocessor files ---
MODEL_PATH = 'laptop_price_model.pkl'
PREPROCESSOR_PATH = 'laptop_preprocessor.pkl'

# --- Utility functions for feature engineering ---
def extract_screen_type(resolution_str):
    types = []
    if 'IPS Panel' in resolution_str:
        types.append('IPS Panel')
    if 'Full HD' in resolution_str:
        types.append('Full HD')
    if 'Retina Display' in resolution_str:
        types.append('Retina Display')
    if 'Touchscreen' in resolution_str:
        types.append('Touchscreen')
    return ', '.join(types) if types else 'Other'

def extract_cpu_brand(cpu_string):
    return cpu_string.split()[0]

def extract_processing_level(cpu_string):
    patterns = [r'Core i[357]', r'Celeron', r'Pentium', r'AMD']
    for pattern in patterns:
        match = re.search(pattern, cpu_string)
        if match:
            return match.group(0)
    return 'Other'

def extract_cpu_details(cpu_string):
    parts = cpu_string.split()
    if len(parts) > 2:
        return ' '.join(parts[2:])
    elif len(parts) > 1:
        return ' '.join(parts[1:])
    return ''

def extract_memory_value(memory_str):
    match = re.search(r'(\d+)(GB|TB)', memory_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == 'TB':
            value *= 1024
        return value
    return 0

def extract_memory_type(memory_str):
    types = []
    if 'SSD' in memory_str:
        types.append('SSD')
    if 'HDD' in memory_str:
        types.append('HDD')
    if 'Flash Storage' in memory_str:
        types.append('Flash Storage')
    if 'Hybrid' in memory_str:
        types.append('Hybrid')
    return ', '.join(types) if types else 'Other'

def extract_gpu_company(gpu_string):
    return gpu_string.split()[0]

def extract_gpu_type(gpu_string):
    parts = gpu_string.split()
    if len(parts) > 1:
        return ' '.join(parts[1:])
    return ''

# --- Load the model and preprocessor from disk ---
# This is a key change. We load the objects once at the start.
@st.cache_data
def load_assets():
    try:
        # Load the trained model
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

        # Load the preprocessor (label encoders and feature lists)
        with open(PREPROCESSOR_PATH, 'rb') as file:
            preprocessor = pickle.load(file)
            label_encoders = preprocessor['label_encoders']
            categorical_options = preprocessor['categorical_options']
            numerical_ranges = preprocessor['numerical_ranges']
            selected_features_encoded = preprocessor['selected_features_encoded']

        return model, label_encoders, categorical_options, numerical_ranges, selected_features_encoded
    except FileNotFoundError:
        st.error(f"Required files not found. Please ensure both '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model assets: {e}")
        st.stop()

model, label_encoders, categorical_options, numerical_ranges, selected_features_encoded = load_assets()

# --- Streamlit UI for user input ---
st.write('Enter the details of the laptop to predict its price.')
st.header('Enter Laptop Features')

# Input fields for numerical features
cols_numerical = st.columns(3)
with cols_numerical[0]:
    inches = st.number_input('Inches:', min_value=numerical_ranges['Inches']['min'], max_value=numerical_ranges['Inches']['max'], value=numerical_ranges['Inches']['mean'], step=0.1)
with cols_numerical[1]:
    ram = st.number_input('Ram (GB):', min_value=numerical_ranges['Ram']['min'], max_value=numerical_ranges['Ram']['max'], value=numerical_ranges['Ram']['mean'], step=1.0)
with cols_numerical[2]:
    weight = st.number_input('Weight (kg):', min_value=numerical_ranges['Weight']['min'], max_value=numerical_ranges['Weight']['max'], value=numerical_ranges['Weight']['mean'], step=0.01)
memory_value = st.number_input('Memory Value (GB):', min_value=numerical_ranges['Memory_Value']['min'], max_value=numerical_ranges['Memory_Value']['max'], value=numerical_ranges['Memory_Value']['mean'], step=1.0)

st.markdown("---")

# Input fields for categorical features
cols_cat1 = st.columns(3)
with cols_cat1[0]:
    company = st.selectbox('Company:', options=categorical_options['Company'])
with cols_cat1[1]:
    typename = st.selectbox('TypeName:', options=categorical_options['TypeName'])
with cols_cat1[2]:
    opsys = st.selectbox('OpSys:', options=categorical_options['OpSys'])

cols_cat2 = st.columns(3)
with cols_cat2[0]:
    screen_type = st.selectbox('Screen Type:', options=categorical_options['Screen_Type'])
with cols_cat2[1]:
    cpu_brand = st.selectbox('CPU Brand:', options=categorical_options['CPU_Brand'])
with cols_cat2[2]:
    processing_level = st.selectbox('Processing Level:', options=categorical_options['Processing_Level'])

cols_cat3 = st.columns(3)
with cols_cat3[0]:
    cpu_details = st.selectbox('CPU Details:', options=categorical_options['CPU_Details'])
with cols_cat3[1]:
    memory_type = st.selectbox('Memory Type:', options=categorical_options['Memory_Type'])
with cols_cat3[2]:
    gpu_company = st.selectbox('GPU Company:', options=categorical_options['GPU_Company'])

gpu_type = st.selectbox('GPU Type:', options=categorical_options['GPU_Type'])

st.markdown("---")

# Prediction button
if st.button('Predict Price'):
    try:
        # Create a dictionary for the user input
        user_input = {
            'Inches': inches,
            'Ram': ram,
            'Weight': weight,
            'Memory_Value': memory_value,
            'Company': company,
            'TypeName': typename,
            'OpSys': opsys,
            'Screen_Type': screen_type,
            'CPU_Brand': cpu_brand,
            'Processing_Level': processing_level,
            'CPU_Details': cpu_details,
            'Memory_Type': memory_type,
            'GPU_Company': gpu_company,
            'GPU_Type': gpu_type
        }

        # Create a DataFrame from the input
        input_df = pd.DataFrame([user_input])

        # Apply encoding using the loaded label_encoders
        for col, le in label_encoders.items():
            input_df[col + '_Encoded'] = le.transform(input_df[col])

        # Select the features in the correct order for the model
        input_for_prediction = input_df[selected_features_encoded]

        # Make prediction
        predicted_price = model.predict(input_for_prediction)

        # Display the result
        st.success(f'Predicted Price: €{np.exp(predicted_price[0]):,.2f}') # Use np.exp since the model likely predicted log price
        st.balloons()

    except ValueError as ve:
        st.error(f"Input error: {ve}. Please check your selections.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
