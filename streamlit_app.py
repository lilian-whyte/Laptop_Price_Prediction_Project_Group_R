#!/usr/bin/env python
import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Include CSS for improved styling
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
.st-cy { /* Style for input labels */
    color: #E6E6FA; /* Lighter indigo for labels */
}
.st-d1 { /* Style for number input */
    background-color: #6A5ACD; /* Slate blue background for input fields */
    color: #F0F2F6; /* Light text color for input fields */
    border-radius: 5px;
    padding: 10px;
}
.st-ch { /* Style for selectbox */
    background-color: #6A5ACD; /* Slate blue background for selectbox */
    color: #F0F2F6; /* Light text color for selectbox */
    border-radius: 5px;
    padding: 10px;
}
.stButton > button { /* Style for the button */
    background-color: #8A2BE2; /* Blueviolet button background */
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton > button:hover { /* Style for the button on hover */
    background-color: #9370DB; /* Mediumpurple on hover */
    color: white;
}
h1, h2, h3, h4, h5, h6 { /* Style for headers */
    color: #E6E6FA; /* Lighter indigo for headers */
}
</style>
""", unsafe_allow_html=True)


import pandas as pd
import streamlit as st

# ... other imports and code

st.title("Laptop Price Prediction")

# This is the original sharing link you get from Google Drive
share_link = "https://drive.google.com/file/d/1POVnnp6fBP97E-bCev1s-BnPqMENGzv8/view?usp=drive_link"

# Extract the file ID from the sharing link
try:
    file_id = share_link.split('/')[-2]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
except IndexError:
    st.error("Invalid Google Drive sharing link format.")
    st.stop()

# Read the CSV file from the constructed URL
try:
    df_original = pd.read_csv(url, encoding='latin-1')
    st.success("CSV file loaded successfully!")
    st.dataframe(df_original.head()) # Optional: show the first few rows
except Exception as e:
    st.error(f"Error loading CSV from Google Drive: {e}")
    st.info("Please ensure the Google Drive file is publicly shared.")
    st.stop()

# ... the rest of your app logic


# Set the title of the web application
st.title('Laptop Price Prediction')

# Add a welcoming message
st.write('Enter the details of the laptop to predict its price.')

# Add input fields for the features
st.header('Enter Laptop Features')

# Define the selected features used for training (using original names for user input)
selected_features_user = [
    'Inches',
    'Ram',
    'Weight',
    'Company',
    'TypeName',
    'Screen_Type',
    'CPU_Brand',
    'Processing_Level',
    'CPU_Details',
    'Memory_Value',
    'Memory_Type',
    'GPU_Company',
    'GPU_Type',
    'OpSys'
]

# --- Feature Engineering Functions (Copied from Notebook) ---

def extract_dimensions(resolution_str):
    """Extracts screen dimensions from a resolution string."""
    match = re.search(r'(\d+x\d+)', resolution_str)
    if match:
        return match.group(1)
    return None

def extract_screen_type(resolution_str):
    """Extracts screen types from a resolution string."""
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
    """Extracts the CPU brand (first word) from a CPU string."""
    return cpu_string.split()[0]

def extract_processing_level(cpu_string):
    """Extracts the processing level from a CPU string."""
    # Look for patterns like 'Core i3', 'Core i5', 'Core i7', 'Celeron', 'Pentium', 'AMD'
    patterns = [r'Core i[357]', r'Celeron', r'Pentium', r'AMD']
    for pattern in patterns:
        match = re.search(pattern, cpu_string)
        if match:
            return match.group(0)
    return 'Other'

def extract_cpu_details(cpu_string):
    """Extracts CPU details (clock speed, model number) excluding brand and processing level."""
    # Remove brand and processing level (assuming they are the first words)
    parts = cpu_string.split()
    # Join the remaining parts, skipping the brand and processing level
    if len(parts) > 2:
        return ' '.join(parts[2:])
    elif len(parts) > 1:
        return ' '.join(parts[1:])
    return ''

def extract_memory_value(memory_str):
    """Extracts the numerical memory value from a memory string."""
    match = re.search(r'(\d+)(GB|TB)', memory_str)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == 'TB':
            value *= 1024  # Convert TB to GB
        return value
    return 0  # Return 0 for unhandled cases or missing values

def extract_memory_type(memory_str):
    """Extracts the memory type from a memory string."""
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
    """Extracts the GPU company (first word) from a GPU string."""
    return gpu_string.split()[0]

def extract_gpu_type(gpu_string):
    """Extracts the GPU type (parts after the first word) from a GPU string."""
    parts = gpu_string.split()
    if len(parts) > 1:
        return ' '.join(parts[1:])
    return ''

def extract_gpu_details(gpu_string):
    """Extracts the numerical or specific model details following the type from a GPU string."""
    parts = gpu_string.split()
    # Check if there are more than two parts (company and type)
    if len(parts) > 2:
        # Join the parts from the third part onwards
        return ' '.join(parts[2:])
    # If there are two or fewer parts, there are no further details to extract
    return ''


# --- Apply Feature Engineering and Encoding to Original Data for Mappings ---

# Create temporary columns for feature extraction to get unique values for selectboxes
df_original['Screen_Dimensions'] = df_original['ScreenResolution'].apply(extract_dimensions)
df_original['Screen_Type'] = df_original['ScreenResolution'].apply(extract_screen_type)
df_original['CPU_Brand'] = df_original['Cpu'].apply(extract_cpu_brand)
df_original['Processing_Level'] = df_original['Cpu'].apply(extract_processing_level)
df_original['CPU_Details'] = df_original['Cpu'].apply(extract_cpu_details)
df_original['Memory_Value'] = df_original['Memory'].apply(extract_memory_value)
df_original['Memory_Type'] = df_original['Memory'].apply(extract_memory_type)
df_original['GPU_Company'] = df_original['Gpu'].apply(extract_gpu_company)
df_original['GPU_Type'] = df_original['Gpu'].apply(extract_gpu_type)
df_original['GPU_Details'] = df_original['Gpu'].apply(extract_gpu_details)

# Convert 'Weight' and 'Ram' to numeric types
df_original['Weight'] = df_original['Weight'].str.replace('kg', '').astype(float)
df_original['Ram'] = df_original['Ram'].str.replace('GB', '').astype(int)


# Create LabelEncoders and fit them on the original data for consistent encoding
label_encoders = {}
categorical_cols = ['Company', 'TypeName', 'Screen_Type', 'CPU_Brand', 'Processing_Level', 'CPU_Details', 'Memory_Type', 'GPU_Company', 'GPU_Type', 'OpSys']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(df_original[col])

# Load the trained model
import pickle
from sklearn.ensemble import RandomForestRegressor
# Assuming 'model' is your trained RandomForestRegressor object
# This code will create a file named 'model.pkl' in the current directory
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('models.pkl', 'rb') as f:
    model = pickle.load(f)
# --- Streamlit UI ---

# Input fields based on selected_features_user
input_data = {}

for feature in selected_features_user:
    if feature in ['Inches', 'Ram', 'Weight', 'Memory_Value']:
        # Numerical input
        min_val = float(df_original[feature].min())
        max_val = float(df_original[feature].max())
        mean_val = float(df_original[feature].mean())
        input_data[feature] = st.number_input(f'{feature}:', min_value=min_val, max_value=max_val, value=mean_val, step=0.1)
    elif feature in ['Company', 'TypeName', 'Screen_Type', 'CPU_Brand', 'Processing_Level', 'CPU_Details', 'Memory_Type', 'GPU_Company', 'GPU_Type', 'OpSys']:
        # Categorical input using selectbox
        options = df_original[feature].unique().tolist()
        options.sort() # Sort options for better user experience
        input_data[feature] = st.selectbox(f'{feature}:', options)


# Add a button to trigger prediction
if st.button('Predict Price'):
    # --- Prediction Logic ---

    # Process input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the same feature engineering and encoding steps to the input data
    # based on the user's selections in the selectboxes.

    # Encoding categorical features
    for col in categorical_cols:
        # Use the fitted label encoders
        input_df[col + '_Encoded'] = label_encoders[col].transform(input_df[col])

    # Select the features in the order used during training
    # Ensure the column names match the training data (encoded names for categorical)
    selected_features_encoded = [
        'Inches',
        'Ram',
        'Weight',
        'Company_Encoded',
        'TypeName_Encoded',
        'Screen_Type_Encoded',
        'CPU_Brand_Encoded',
        'Processing_Level_Encoded',
        'CPU_Details_Encoded',
        'Memory_Value',
        'Memory_Type_Encoded',
        'GPU_Company_Encoded',
        'GPU_Type_Encoded',
        'OpSys_Encoded'
    ]

    # Create the final input array for prediction
    # We only need the encoded categorical features and the numerical features
    input_for_prediction = input_df[selected_features_encoded]

    # Make prediction
    predicted_price = model.predict(input_for_prediction)

    # Display the predicted price
    st.subheader(f'Predicted Price: €{predicted_price[0]:.2f}')



