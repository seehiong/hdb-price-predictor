# app.py
import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(layout="centered", page_title="HDB Price Predictor")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Configuration ---
KSERVE_URL = os.environ.get("KSERVE_URL", "http://140.245.54.38:80/v2/models/hdb-resale-xgb/infer")
KSERVE_HOST = os.environ.get("KSERVE_HOST", "hdb-resale-xgb-kserve-test.example.com")
SCALER_PATH = "scaler.joblib"

# Feature names in the EXACT order the model expects
FEATURE_NAMES = ['floor_area_sqm', 'remaining_lease_years', 'storey_avg', 'sale_year', 'sale_month',
                 'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM',
                 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION', 'flat_model_2-room', 'flat_model_3Gen', 'flat_model_Adjoined flat',
                 'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved', 'flat_model_Improved-Maisonette', 'flat_model_Maisonette',
                 'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2', 'flat_model_Multi Generation', 'flat_model_New Generation',
                 'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft', 'flat_model_Premium Maisonette', 'flat_model_Simplified', 'flat_model_Standard',
                 'flat_model_Terrace', 'flat_model_Type S1', 'flat_model_Type S2', 'town_ANG MO KIO', 'town_BEDOK',
                 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH',
                 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG',
                 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS',
                 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON',
                 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN']

NUM_FEATURES = len(FEATURE_NAMES)

# --- Extract Categories for Dropdowns ---
def extract_categories(prefix, feature_list):
    categories = []
    prefix_len = len(prefix)
    for feature in feature_list:
        if feature.startswith(prefix):
            categories.append(feature[prefix_len:])
    return sorted(categories)

FLAT_TYPES = extract_categories("flat_type_", FEATURE_NAMES)
FLAT_MODELS = extract_categories("flat_model_", FEATURE_NAMES)
TOWNS = extract_categories("town_", FEATURE_NAMES)

# --- Load Scaler ---
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: Scaler file not found at {SCALER_PATH}. Cannot proceed.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("HDB Resale Price Predictor")
st.markdown("Enter the details below to get a price prediction.")

# --- Input Fields ---
col1, col2 = st.columns(2) # Arrange inputs in columns

with col1:
    st.header("Key Features")
    floor_area = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=300.0, value=90.0, step=1.0)
    lease_years = st.number_input("Remaining Lease (Years)", min_value=10.0, max_value=99.0, value=70.0, step=0.5)
    storey = st.number_input("Storey (Average)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

with col2:
    st.header("Location & Type")
    # Set default indices based on common values or simple index 0
    default_town_index = TOWNS.index("TAMPINES") if "TAMPINES" in TOWNS else 0
    default_flat_type_index = FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0
    default_flat_model_index = FLAT_MODELS.index("Improved") if "Improved" in FLAT_MODELS else 0

    selected_town = st.selectbox("Town", TOWNS, index=default_town_index)
    selected_flat_type = st.selectbox("Flat Type", FLAT_TYPES, index=default_flat_type_index)
    selected_flat_model = st.selectbox("Flat Model", FLAT_MODELS, index=default_flat_model_index)


# --- Prediction Button ---
if st.button("Predict Resale Price", type="primary"):
    st.markdown("---")
    st.subheader("Processing...")

    # --- Prepare Input Data ---
    # Start with all features as 0.0
    input_dict = {feat: 0.0 for feat in FEATURE_NAMES}

    # Update with numerical user inputs
    input_dict['floor_area_sqm'] = float(floor_area)
    input_dict['remaining_lease_years'] = float(lease_years)
    input_dict['storey_avg'] = float(storey)

    # Update with default sale year/month
    now = datetime.now()
    input_dict['sale_year'] = float(now.year)
    input_dict['sale_month'] = float(now.month)

    # *** Update one-hot encoded features based on dropdown selections ***
    # Town
    town_feature_name = f"town_{selected_town}"
    if town_feature_name in input_dict:
        input_dict[town_feature_name] = 1.0
    else:
        st.warning(f"Selected town feature '{town_feature_name}' not found in model features. Check FEATURE_NAMES.")

    # Flat Type
    flat_type_feature_name = f"flat_type_{selected_flat_type}"
    if flat_type_feature_name in input_dict:
        input_dict[flat_type_feature_name] = 1.0
    else:
        st.warning(f"Selected flat type feature '{flat_type_feature_name}' not found in model features.")

    # Flat Model
    flat_model_feature_name = f"flat_model_{selected_flat_model}"
    if flat_model_feature_name in input_dict:
        input_dict[flat_model_feature_name] = 1.0
    else:
        st.warning(f"Selected flat model feature '{flat_model_feature_name}' not found in model features.")


    # Convert dictionary to ordered list
    try:
        input_list = [input_dict[feature] for feature in FEATURE_NAMES]
        input_array = np.array(input_list).astype(np.float32).reshape(1, -1) # Reshape for scaler
        st.write(f"Input shape before scaling: {input_array.shape}")
        # Optional: Display which one-hot features are set
        # set_features = {k:v for k,v in input_dict.items() if v==1.0 and ('town_' in k or 'flat_type_' in k or 'flat_model_' in k)}
        # st.write(f"One-hot features set: {set_features}")


        # --- Scale the Input Data ---
        input_scaled = scaler.transform(input_array)
        st.write(f"Input shape after scaling: {input_scaled.shape}")
        payload_data = input_scaled.flatten().tolist()

        # --- Construct KServe Payload ---
        payload = {
            "inputs": [{
                "name": "input-0",
                "shape": [1, NUM_FEATURES],
                "datatype": "FP32",
                "data": payload_data
            }]
        }
        st.write("Sending request to KServe...")

        # --- Send Request ---
        try:
            headers = {"Content-Type": "application/json"}
            if KSERVE_HOST:
                headers["Host"] = KSERVE_HOST

            response = requests.post(KSERVE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            st.write("Response received:")
            prediction = result.get('outputs', [{}])[0].get('data', [None])[0]

            if prediction is not None:
                st.success(f"**Predicted Resale Price:** S$ {prediction:,.2f}")
            else:
                st.error("Prediction data not found in the response.")
                st.json(result)

        # ... (Error handling remains the same) ...
        except requests.exceptions.Timeout:
            st.error(f"Error: Request timed out connecting to {KSERVE_URL}")
        except requests.exceptions.ConnectionError:
            st.error(f"Error: Could not connect to the prediction service at {KSERVE_URL}. Is it running and accessible?")
        except requests.exceptions.HTTPError as err:
            st.error(f"Error: Prediction service returned status code {err.response.status_code}.")
            try:
                error_detail = err.response.json()
                st.json({"error_details": error_detail})
            except json.JSONDecodeError:
                 st.text(f"Response content:\n{err.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        st.error(f"Error preparing data for prediction: {e}")