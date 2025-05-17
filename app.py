# app.py
import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    layout="wide",
    page_title="HDB Resale Price Predictor",
    page_icon="🏙️",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
/* Mobile responsiveness */
@media (max-width: 768px) {
    .stApp {
        padding: 10px;
    }
    .main-header {
        font-size: 24px !important;
    }
    .sub-header {
        font-size: 18px !important;
    }
}
/* Custom styling */
.main-header {
    color: #1E88E5;
    font-size: 36px;
    font-weight: 700;
}
.sub-header {
    color: #0D47A1;
    font-size: 22px;
    font-weight: 600;
}
.info-box {
    background-color: #E3F2FD;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin-bottom: 20px;
}
.prediction-card {
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
    border-left: 5px solid #43A047;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
/* Toggle button styling */
.toggle-container {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.toggle-label {
    margin-right: 10px;
    font-weight: 500;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Configuration ---
KSERVE_URL = "http://140.245.54.38:80/v2/models/hdb-resale-price-xgb/infer"
KSERVE_HOST = "hdb-resale-price-xgb-kserve-test.example.com"
SCALER_PATH = "scaler.joblib"
POSTAL_DATA_PATH = "postal_data.json"

# Model details
MODEL_TYPE = "XGBoost"
DATA_INGESTION_DATE = "14-05-2025"

# Feature names in the EXACT order the model expects
FEATURE_NAMES = ['floor_area_sqm', 'postal', 'storey_avg', 'sale_year', 'sale_month', 'remaining_lease_years', 'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION', 'flat_model_2-ROOM', 'flat_model_3GEN', 'flat_model_ADJOINED FLAT', 'flat_model_APARTMENT', 'flat_model_DBSS', 'flat_model_IMPROVED', 'flat_model_IMPROVED-MAISONETTE', 'flat_model_MAISONETTE', 'flat_model_MODEL A', 'flat_model_MODEL A-MAISONETTE', 'flat_model_MODEL A2', 'flat_model_MULTI GENERATION', 'flat_model_NEW GENERATION', 'flat_model_PREMIUM APARTMENT', 'flat_model_PREMIUM APARTMENT LOFT', 'flat_model_PREMIUM MAISONETTE', 'flat_model_SIMPLIFIED', 'flat_model_STANDARD', 'flat_model_TERRACE', 'flat_model_TYPE S1', 'flat_model_TYPE S2', 'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN']

# Confirm feature count matches training data
if len(FEATURE_NAMES) != 60:
    st.error(f"Feature count mismatch! Expected 60 but got {len(FEATURE_NAMES)}")

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

# --- Generate past / future years and months ---
current_date = datetime.now()
FUTURE_YEARS = list(range(current_date.year - 10, current_date.year + 10))
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_TO_NUM = {month: i+1 for i, month in enumerate(MONTHS)}

# --- Unit conversion function ---
def sqm_to_sqft(sqm):
    return sqm * 10.7639

def sqft_to_sqm(sqft):
    return sqft / 10.7639

# --- Load Scaler ---
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
        return scaler
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {SCALER_PATH}. Cannot proceed.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

# --- Load Postal Data ---
@st.cache_resource
def load_postal_data():
    try:
        with open(POSTAL_DATA_PATH, 'r') as f:
            postal_data = json.load(f)
        print(f"Postal data loaded successfully with {len(postal_data)} entries.")
        return postal_data
    except FileNotFoundError:
        st.error(f"Error: Postal data file not found at {POSTAL_DATA_PATH}.")
        return {}
    except Exception as e:
        st.error(f"Error loading postal data: {e}")
        return {}

scaler = load_scaler()
postal_data = load_postal_data()

# --- Validate Postal Code ---
def validate_postal_code(postal_code):
    # Basic validation
    if not postal_code or not postal_code.isdigit() or len(postal_code) != 6:
        return None, "Please enter a valid 6-digit postal code."
    
    # Check if postal code exists in our database
    if postal_code not in postal_data:
        return None, f"Postal code {postal_code} not found in our database."
    
    # Get the first entry
    postal_info = postal_data[postal_code][0]
    return postal_info, None

# --- Initialize Session State Variables ---
if 'selected_town' not in st.session_state:
    st.session_state.selected_town = None
if 'lease_commencement_year' not in st.session_state:
    st.session_state.lease_commencement_year = 1966
if 'postal_validation_error' not in st.session_state:
    st.session_state.postal_validation_error = None

# --- Streamlit UI ---
st.markdown('<h1 class="main-header">HDB Resale Price Predictor</h1>', unsafe_allow_html=True)

with st.container():
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Model:</strong> {MODEL_TYPE} | <strong>Data Last Updated:</strong> {DATA_INGESTION_DATE}</p>
        <p>This predictor estimates HDB resale prices based on historical transaction data.</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Make Prediction", "📍 View Transaction Map", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">Property Details</h2>', unsafe_allow_html=True)

        use_sqft = st.toggle("Use Square Feet (sqft) instead of Square Meters (sqm)", value=False, key="sqft_toggle")

        if use_sqft:
            floor_area_sqft = st.number_input(
                "Floor Area (sqft)",
                min_value=215.0,  # ~20 sqm
                max_value=3230.0,  # ~300 sqm
                value=969.0,  # ~90 sqm
                step=10.0
            )
            floor_area = sqft_to_sqm(floor_area_sqft)
        else:
            floor_area = st.number_input(
                "Floor Area (sqm)",
                min_value=20.0,
                max_value=300.0,
                value=90.0,
                step=1.0
            )

        # Lease commencement year will be updated based on postal code, if available
        lease_commencement_year = st.number_input(
            "Lease Commencement Year",
            min_value=1966,
            max_value=current_date.year,
            value=st.session_state.lease_commencement_year,
            step=1,
            help="Enter the year the flat's 99-year lease started.",
            key="lease_year_input"
        )

        storey = st.number_input(
            "Storey (Average)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=1.0
        )

        st.markdown("##### Expected Sale Date")
        col_year, col_month = st.columns(2)
        with col_year:
            selected_year = st.selectbox("Year", FUTURE_YEARS, index=10)
        with col_month:
            selected_month = st.selectbox("Month", MONTHS, index=current_date.month+1)

    with col2:
        st.markdown('<h2 class="sub-header">Location & Type</h2>', unsafe_allow_html=True)

        use_postal = st.toggle("Enter Postal Code Instead of Town", value=False, key="postal_toggle")
        postal_error_container = st.empty()
        
        if use_postal:
            postal_code = st.text_input(
                "Postal Code",
                placeholder="e.g., 760123",
                help="Enter the 6-digit Singapore postal code",
                key="postal_code_input"
            )
            
            if postal_code:
                postal_info, error_msg = validate_postal_code(postal_code)
                if error_msg:
                    postal_error_container.error(error_msg)
                    st.session_state.postal_validation_error = error_msg
                    st.session_state.selected_town = None
                else:
                    postal_error_container.success(f"Postal code validated: {postal_code}")
                    st.session_state.postal_validation_error = None
                    
                    if postal_info:
                        st.session_state.selected_town = postal_info["town"]
                        st.session_state.lease_commencement_year = int(postal_info["lease_commence_date"])
                        
                        st.info(f"Auto-filled from postal code: Town: {postal_info['town']}, Lease Year: {postal_info['lease_commence_date']}")
            
            if st.session_state.selected_town:
                town_index = TOWNS.index(st.session_state.selected_town) if st.session_state.selected_town in TOWNS else 0
                st.selectbox(
                    "Town (Auto-filled from postal code)",
                    TOWNS,
                    index=town_index,
                    disabled=True,
                    key="town_disabled"
                )
            else:
                st.selectbox(
                    "Town (Will be auto-filled when postal code is validated)",
                    TOWNS,
                    disabled=True,
                    key="town_disabled_empty"
                )
        else:
            default_town_index = TOWNS.index("TAMPINES") if "TAMPINES" in TOWNS else 0
            selected_town = st.selectbox(
                "Town",
                TOWNS,
                index=default_town_index,
                help="Select the HDB township",
                key="town_active"
            )  
            st.session_state.selected_town = selected_town
            postal_code = None

        default_flat_type_index = FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0
        default_flat_model_index = FLAT_MODELS.index("IMPROVED") if "IMPROVED" in FLAT_MODELS else 0

        selected_flat_type = st.selectbox(
            "Flat Type",
            FLAT_TYPES,
            index=default_flat_type_index,
            help="Select the flat type (e.g., 3 ROOM, 4 ROOM)"
        )

        selected_flat_model = st.selectbox(
            "Flat Model",
            FLAT_MODELS,
            index=default_flat_model_index,
            help="Select the specific model of the flat"
        )

    st.markdown("")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🔮 Predict Resale Price", type="primary", use_container_width=True)

    if predict_button:
        if use_postal and st.session_state.postal_validation_error:
            st.error("Please enter a valid postal code before making a prediction.")
        elif use_postal and not postal_code:
            st.error("Please enter a postal code before making a prediction.")
        else:
            with st.spinner("Analyzing market data..."):
                # --- Calculate Remaining Lease ---
                lease_commencement_year_value = st.session_state.lease_commencement_year if use_postal else lease_commencement_year
                lease_duration_at_sale = float(selected_year) - float(lease_commencement_year_value)
                calculated_remaining_lease_years = 99.0 - lease_duration_at_sale
                
                model_trained_min_lease = 10.0
                model_trained_max_lease = 99.0
                calculated_remaining_lease_years = np.clip(
                    calculated_remaining_lease_years,
                    model_trained_min_lease,
                    model_trained_max_lease
                )
        
                input_values = {}
                input_values['floor_area_sqm'] = float(floor_area)
                input_values['postal'] = float(postal_code) if postal_code else 0.0  
                input_values['remaining_lease_years'] = float(calculated_remaining_lease_years)
                input_values['storey_avg'] = float(storey)
                input_values['sale_year'] = float(selected_year)
                input_values['sale_month'] = float(MONTH_TO_NUM[selected_month])

                final_input_for_model = {feat: 0.0 for feat in FEATURE_NAMES}

                for key, value in input_values.items():
                    if key in final_input_for_model:
                        final_input_for_model[key] = value

                town_to_use = st.session_state.selected_town if use_postal else selected_town
                
                town_feature_name = f"town_{town_to_use}"
                if town_feature_name in final_input_for_model:
                    final_input_for_model[town_feature_name] = 1.0
                else:
                    st.warning(f"Selected town '{town_to_use}' (feature: {town_feature_name}) not in model's FEATURE_NAMES. Check for typos or data mismatches.")

                flat_type_feature_name = f"flat_type_{selected_flat_type}"
                if flat_type_feature_name in final_input_for_model:
                    final_input_for_model[flat_type_feature_name] = 1.0
                else:
                    st.warning(f"Selected flat type '{selected_flat_type}' (feature: {flat_type_feature_name}) not in model's FEATURE_NAMES.")

                flat_model_feature_name = f"flat_model_{selected_flat_model}"
                if flat_model_feature_name in final_input_for_model:
                    final_input_for_model[flat_model_feature_name] = 1.0
                else:
                    st.warning(f"Selected flat model '{selected_flat_model}' (feature: {flat_model_feature_name}) not in model's FEATURE_NAMES.")

                try:
                    input_list = [final_input_for_model[feature] for feature in FEATURE_NAMES]
                    input_df =  np.array(input_list).astype(np.float32).reshape(1, -1)

                    input_scaled = scaler.transform(input_df)
                    payload_data = input_scaled.flatten().tolist()

                    payload = {
                        "inputs": [{
                            "name": "input-0",
                            "shape": [1, NUM_FEATURES],
                            "datatype": "FP32",
                            "data": payload_data
                        }]
                    }

                    try:
                        headers = {"Content-Type": "application/json"}
                        if KSERVE_HOST:
                            headers["Host"] = KSERVE_HOST

                        response = requests.post(KSERVE_URL, headers=headers, json=payload, timeout=30)
                        response.raise_for_status()
                        result = response.json()
                        prediction = result.get('outputs', [{}])[0].get('data', [None])[0]

                        if prediction is not None:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h2>Predicted Resale Price</h2>
                                <h1 style="font-size: 42px; color: #388E3C;">S$ {prediction:,.2f}</h1>
                                <p>For {selected_flat_type} in {town_to_use} (Expected: {selected_month} {selected_year})</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                            with st.expander("View Property Details Summary"):
                                col_details1, col_details2 = st.columns(2)
                                with col_details1:
                                    st.write("**Property Details:**")
                                    if use_sqft:
                                        st.write(f"• Floor Area: {floor_area_sqft:.1f} sqft ({floor_area:.1f} sqm)")
                                    else:
                                        st.write(f"• Floor Area: {floor_area:.1f} sqm ({sqm_to_sqft(floor_area):.1f} sqft)")
                                    st.write(f"• Lease Commencement Year: {lease_commencement_year_value}")
                                    st.write(f"• Remaining Lease (at sale): {calculated_remaining_lease_years:.1f} years")
                                    st.write(f"• Storey (Average): {storey:.1f}")
                    
                                with col_details2:
                                    st.write("**Unit Details:**")
                                    st.write(f"• Town: {town_to_use}")
                                    if postal_code:
                                        st.write(f"• Postal Code: {postal_code}")
                                    st.write(f"• Flat Type: {selected_flat_type}")
                                    st.write(f"• Flat Model: {selected_flat_model}")
                        else:
                            st.error("Prediction data not found in the response.")

                    except requests.exceptions.Timeout:
                        st.error(f"Request timed out. The prediction service may be experiencing high demand.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to the prediction service. Please try again later.")
                    except requests.exceptions.HTTPError as err:
                        st.error(f"Error: Prediction service returned status code {err.response.status_code}.")
                        st.error(f"Response content: {err.response.text}") # More detailed error
                    except Exception as e:
                        st.error(f"An unexpected error occurred during prediction: {str(e)}")

                except Exception as e:
                    st.error(f"Error preparing data for prediction: {str(e)}")

with tab2:
    st.markdown('<h2 class="sub-header">HDB Resale Transaction Map</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p class="info-box">
            Click on any postal code marker to see the last 20 transactions for that location.
        </p>
        """, unsafe_allow_html=True)
    
    st.components.v1.iframe(
        src="https://axmbarc6hpai.objectstorage.ap-singapore-2.oci.customer-oci.com/n/axmbarc6hpai/b/nn-bucket/o/hdb_model_for_kserve%2Fhdb_resale_price_map_clickable.html",
        height=800,
        width=1200,
        scrolling=True
    )
    
with tab3:
    st.markdown("""
    ## About This Predictor

    This HDB resale price predictor uses a machine learning model to estimate the potential selling price of an HDB flat based on its characteristics and location.

    ### Model Information
    - **Model Type**: XGBoost Regression Model
    - **Data Last Updated**: {}
    - **Features Used**: Floor area, remaining lease (calculated from lease commencement year and sale date), storey, flat type, flat model, and location.

    ### How to Use
    1. Enter your property details in the form, including the **Lease Commencement Year**.
    2. Select the expected sale date (year and month).
    3. Click "Predict Resale Price" to get an estimate.
    
    ### Postal Code Lookup
    - Toggle "Enter Postal Code Instead of Town" to automatically retrieve town and lease commencement information.
    - The system will validate the postal code against our database and auto-fill relevant details.

    ### Important Notes
    - This tool provides estimates only and should not be considered as financial advice.
    - Actual prices may vary based on market conditions and property specifics.
    - The model is trained on historical data and may not capture sudden market changes.
    """.format(DATA_INGESTION_DATE))