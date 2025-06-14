# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# XGBoost import
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning, message=".*Calling st.rerun().*")

st.set_page_config(
    layout="wide",
    page_title="HDB Resale Price Predictor",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

PAGE_OPTIONS_LIST = ["Make Prediction", "Transaction Map", "About"]
DATA_INGESTION_DATE = "16-06-2025"

if 'active_page' not in st.session_state:
    st.session_state.active_page = PAGE_OPTIONS_LIST[0]

if 'menu_key_counter' not in st.session_state:
    st.session_state.menu_key_counter = 0

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

current_theme = st.session_state.theme
theme_colors = {
    'light': {
        'bg': '#FFFFFF', 'text': '#000000', 'primary': '#1E88E5', 'secondary': '#0D47A1',
        'success': '#43A047', 'info_bg': '#E3F2FD', 'info_text': '#000000',
        'info_border': '#1E88E5', 'prediction_bg': '#E8F5E9', 'prediction_border': '#43A047',
        'sidebar_bg': '#F0F2F6', 'sidebar_text': '#000000'
    },
    'dark': {
        'bg': '#121212', 'text': '#FFFFFF', 'primary': '#90CAF9', 'secondary': '#64B5F6',
        'success': '#81C784', 'info_bg': '#0A1929', 'info_text': '#E1E1E1',
        'info_border': '#90CAF9', 'prediction_bg': '#0A2018', 'prediction_border': '#81C784',
        'sidebar_bg': '#1E1E1E', 'sidebar_text': '#FFFFFF'
    }
}
colors = theme_colors[current_theme]

hide_streamlit_style = f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.stApp {{
    max-width: 1200px;
    margin: 0 auto;
    background-color: {colors['bg']};
    color: {colors['text']};
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: {colors['sidebar_bg']};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{ 
    color: {colors['sidebar_text']} !important;
}}
[data-testid="stSidebar"] .stButton > button {{ 
    color: {colors['sidebar_text']} !important; 
}}
[data-testid="stSidebar"] div[data-testid="stToggle"] span {{
    color: {colors['sidebar_text']} !important;
}}
[data-testid="stSidebar"] div[data-testid="stToggle"] label {{
    color: {colors['sidebar_text']} !important;
}}

@media (max-width: 768px) {{
    .stApp {{ padding: 10px; }}
    .main-header {{ font-size: 24px !important; }}
    .sub-header {{ font-size: 18px !important; }}
}}
.main-header {{ color: {colors['primary']}; font-size: 36px; font-weight: 700; }}
.sub-header {{ color: {colors['secondary']}; font-size: 22px; font-weight: 600; }}
.info-box {{
    background-color: {colors['info_bg']}; color: {colors['info_text']};
    padding: 15px; border-radius: 10px; border-left: 5px solid {colors['info_border']};
    margin-bottom: 20px;
}}
.prediction-card {{
    background-color: {colors['prediction_bg']}; padding: 20px; border-radius: 10px;
    text-align: center; margin: 20px 0; border-left: 5px solid {colors['prediction_border']};
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: {colors['text']};
}}
.stMarkdown, body {{ color: {colors['text']} !important; }}

/* Labels for input widgets */
.stTextInput label, 
.stNumberInput label, 
.stSelectbox label, 
.stDateInput label,
.stTimeInput label,
.stMultiSelect label,
.stTextArea label,
.stRadio label,
.stCheckbox label {{
    color: {colors['text']} !important;
}}
div[data-testid="stCheckbox"] label div[data-testid="stMarkdownContainer"] p {{
    color: {colors['text']} !important;
}}

.stTextInput div[data-baseweb="input"] > input,
.stNumberInput div[data-baseweb="input"] > input,
.stTextArea div[data-baseweb="input"] > textarea {{
    color: {colors['text']} !important;
    background-color: {colors['bg']} !important; 
    border: 1px solid {colors['secondary']} !important; 
}}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
    color: {colors['text']} !important;
    background-color: {colors['bg']} !important;
    border: 1px solid {colors['secondary']} !important;
}}
div[data-baseweb="popover"] ul li {{
    background-color: {colors['bg']} !important;
    color: {colors['text']} !important;
}}
div[data-baseweb="popover"] ul li:hover {{
    background-color: {colors['secondary']} !important;
    color: {'#FFFFFF' if current_theme == 'dark' else '#000000'} !important;
}}
.stButton > button {{
    color: {'#FFFFFF' if colors['primary'] not in ['#FFFFFF', '#E1E1E1', '#f0f2f6'] else '#000000'} !important; 
    background-color: {colors['primary']} !important;
    border: 1px solid {colors['primary']} !important;
}}
.stButton > button:hover {{
    background-color: {colors['secondary']} !important;
    border: 1px solid {colors['secondary']} !important;
}}
.stButton > button:focus {{
    box-shadow: 0 0 0 0.2rem {colors['primary']}40 !important;
}}
div[data-testid="stOptionMenu"] button {{
    color: {colors['text']} !important; 
    border-bottom: 2px solid transparent; 
    border-radius: 0 !important; 
    margin-right: 2px; 
    padding: 10px 15px;
}}
div[data-testid="stOptionMenu"] button:hover {{
    color: {colors['primary']} !important; 
    background-color: transparent !important; 
}}
div[data-testid="stOptionMenu"] button[aria-selected="true"] {{
    color: {colors['primary']} !important; 
    border-bottom-color: {colors['primary']} !important; 
    font-weight: bold;
}}
@media (max-width: 768px) {{
    .stApp {{ padding: 10px; }}
    .main-header {{ font-size: 24px !important; }}
    .sub-header {{ font-size: 18px !important; }}
    .mobile-return-button-container {{
        position: fixed; bottom: 20px; right: 20px; z-index: 10000;
    }}
    .mobile-return-button-container div[data-testid="stButton"] > button {{
        border-radius: 50%; width: 50px; height: 50px; padding: 0; font-size: 22px; 
        line-height: 50px; text-align: center; 
        background-color: {colors['primary']} !important; color: white !important; 
        border: none; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    .mobile-return-button-container div[data-testid="stButton"] > button:hover {{
        background-color: {colors['secondary']} !important;
    }}
    div[data-testid="stOptionMenu"] {{
        display: flex !important;
        flex-wrap: wrap !important; 
        justify-content: space-around !important; 
        width: 100% !important;
        padding-bottom: 5px;
    }}
    div[data-testid="stOptionMenu"] button {{
        font-size: 0.8rem !important;
        padding: 8px 5px !important;
        margin: 2px !important; 
        flex-grow: 1;
        flex-basis: auto;
        min-width: calc(33% - 10px);
        text-align: center;
        line-height: 1.2;
    }}
}}
@media (min-width: 769px) {{
    .mobile-return-button-container {{ display: none; }}
}}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Configuration & Helper Functions ---
SCALER_PATH = "scaler.joblib"
POSTAL_DATA_PATH = "postal_data.json"
MODEL_LOCAL_PATH = "model.bst"
MODEL_TYPE = "XGBoost (Locally Loaded)"
FEATURE_NAMES = ['floor_area_sqm', 'postal', 'storey_avg', 'sale_year', 'sale_month', 'remaining_lease_years', 'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION', 'flat_model_2-ROOM', 'flat_model_3GEN', 'flat_model_ADJOINED FLAT', 'flat_model_APARTMENT', 'flat_model_DBSS', 'flat_model_IMPROVED', 'flat_model_IMPROVED-MAISONETTE', 'flat_model_MAISONETTE', 'flat_model_MODEL A', 'flat_model_MODEL A-MAISONETTE', 'flat_model_MODEL A2', 'flat_model_MULTI GENERATION', 'flat_model_NEW GENERATION', 'flat_model_PREMIUM APARTMENT', 'flat_model_PREMIUM APARTMENT LOFT', 'flat_model_PREMIUM MAISONETTE', 'flat_model_SIMPLIFIED', 'flat_model_STANDARD', 'flat_model_TERRACE', 'flat_model_TYPE S1', 'flat_model_TYPE S2', 'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN']

if len(FEATURE_NAMES) != 60:
    st.error(f"Feature count mismatch! Expected 60 but got {len(FEATURE_NAMES)}")
NUM_FEATURES = len(FEATURE_NAMES)


@st.cache_resource
def load_xgboost_model_local():
    try:
        if not os.path.exists(MODEL_LOCAL_PATH):
            st.error(f"Model file not found at {MODEL_LOCAL_PATH}. Please ensure it's in the same directory as app.py.")
            return None
        
        xgb_model = xgb.Booster()
        xgb_model.load_model(MODEL_LOCAL_PATH)
        
        # Optional: Debug model features
        try:
            num_model_features = xgb_model.num_features()
            if num_model_features != NUM_FEATURES:
                st.warning(f"Model feature count ({num_model_features}) does not match expected ({NUM_FEATURES}). This WILL cause errors.")
        except Exception as e_feat:
            st.warning(f"DEBUG: Could not get feature count from model: {e_feat}")

        return xgb_model
    except Exception as e:
        st.error(f"Error loading local XGBoost model ({MODEL_LOCAL_PATH}): {e}")
        return None

def extract_categories(prefix, feature_list):
    categories = [feature[len(prefix):] for feature in feature_list if feature.startswith(prefix)]
    return sorted(categories)

FLAT_TYPES = extract_categories("flat_type_", FEATURE_NAMES)
FLAT_MODELS = extract_categories("flat_model_", FEATURE_NAMES)
TOWNS = extract_categories("town_", FEATURE_NAMES)

current_date = datetime.now()
FUTURE_YEARS = list(range(current_date.year - 10, current_date.year + 10))
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_TO_NUM = {month: i + 1 for i, month in enumerate(MONTHS)}

def sqm_to_sqft(sqm): return sqm * 10.7639
def sqft_to_sqm(sqft): return sqft / 10.7639

@st.cache_resource
def load_scaler():
    try:
        if not os.path.exists(SCALER_PATH):
            st.error(f"Scaler file not found at {SCALER_PATH}. Please ensure it's in the same directory as app.py.")
            st.stop() 
        return joblib.load(SCALER_PATH)
    except Exception as e: st.error(f"Error loading scaler ({SCALER_PATH}): {e}"); st.stop()

@st.cache_resource
def load_postal_data():
    try:
        if not os.path.exists(POSTAL_DATA_PATH):
            st.error(f"Postal data file not found at {POSTAL_DATA_PATH}. Please ensure it's in the same directory as app.py.")
            return {} 
        with open(POSTAL_DATA_PATH, 'r') as f: return json.load(f)
    except Exception as e: st.error(f"Error loading postal data ({POSTAL_DATA_PATH}): {e}"); return {}

# --- Load resources ---
scaler = load_scaler()
postal_data = load_postal_data()
xgb_model_loaded = load_xgboost_model_local() 

def validate_postal_code(postal_code):
    if not postal_code or not postal_code.isdigit() or len(postal_code) != 6:
        return None, "Please enter a valid 6-digit postal code."
    if not postal_data:
        return None, "Postal code database not loaded. Cannot validate."
    if postal_code not in postal_data:
        return None, f"Postal code {postal_code} not found in our database."
    return postal_data[postal_code][0], None

if 'selected_town' not in st.session_state: st.session_state.selected_town = None
if 'lease_commencement_year' not in st.session_state: st.session_state.lease_commencement_year = 1966
if 'postal_validation_error' not in st.session_state: st.session_state.postal_validation_error = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("### App Settings")
    st.toggle(
        "Dark Mode",
        value=st.session_state.theme == 'dark',
        on_change=toggle_theme,
        key="theme_toggle_sidebar"
    )
    if st.button("🏠 Return to Predictor"):
        st.session_state.active_page = PAGE_OPTIONS_LIST[0]
        st.session_state.menu_key_counter += 1
        st.rerun()

# --- Main App UI ---
st.markdown('<h1 class="main-header">HDB Resale Price Predictor</h1>', unsafe_allow_html=True)
with st.container():
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Model:</strong> {MODEL_TYPE} | <strong>Data Last Updated:</strong> {DATA_INGESTION_DATE}</p>
        <p><strong>Powered by AI, this cutting-edge predictor leverages an XGBoost model trained on extensive historical resale transaction records to estimate HDB resale prices with precision.</strong></p>
    </div>
    """, unsafe_allow_html=True)

icons_list = ['pencil-square', 'geo-alt-fill', 'info-circle-fill']
try:
    default_nav_index = PAGE_OPTIONS_LIST.index(st.session_state.active_page)
except ValueError:
    default_nav_index = 0
    st.session_state.active_page = PAGE_OPTIONS_LIST[0]

current_menu_key = f"main_nav_menu_{st.session_state.menu_key_counter}"

selected_page_from_menu = option_menu(
    menu_title=None,
    options=PAGE_OPTIONS_LIST,
    icons=icons_list,
    menu_icon="cast",
    default_index=default_nav_index,
    orientation="horizontal",
    key=current_menu_key
)

if selected_page_from_menu != st.session_state.active_page:
    st.session_state.active_page = selected_page_from_menu
    st.rerun()

if st.session_state.active_page == PAGE_OPTIONS_LIST[0]: # "Make Prediction"

    st.markdown('<h2 class="sub-header">Expected Sale Date</h2>', unsafe_allow_html=True)
    sale_date_col1, sale_date_col2 = st.columns(2) 
    with sale_date_col1:
        try:
            default_year_idx = FUTURE_YEARS.index(current_date.year)
        except ValueError:
            default_year_idx = len(FUTURE_YEARS) // 2
        selected_year = st.selectbox("Year", FUTURE_YEARS, index=default_year_idx, key="sale_year_select") 
    with sale_date_col2:
        selected_month = st.selectbox("Month", MONTHS, index=current_date.month -1, key="sale_month_select") 
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<h2 class="sub-header">Property Details</h2>', unsafe_allow_html=True)
        use_sqft = st.toggle("Use Square Feet (sqft)", value=False, key="sqft_toggle")
        if use_sqft:
            floor_area_sqft = st.number_input("Floor Area (sqft)", min_value=215.0, max_value=3230.0, value=969.0, step=10.0)
            floor_area = sqft_to_sqm(floor_area_sqft)
        else:
            floor_area = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=300.0, value=90.0, step=1.0)

        lease_commencement_year = st.number_input(
            "Lease Commencement Year", min_value=1966, max_value=current_date.year,
            value=st.session_state.lease_commencement_year, step=1,
            help="Enter the year the flat's 99-year lease started.", key="lease_year_input"
        )
        storey = st.number_input("Storey (Average)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

    with col2:
        st.markdown('<h2 class="sub-header">Location & Type</h2>', unsafe_allow_html=True)
        use_postal = st.toggle("Enter Postal Code Instead of Town", value=False, key="postal_toggle")
        postal_error_container = st.empty()
        
        if use_postal:
            postal_code = st.text_input("Postal Code", placeholder="e.g., 760123", key="postal_code_input")
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
                        town_changed = st.session_state.selected_town != postal_info["town"]
                        lease_year_changed = st.session_state.lease_commencement_year != int(postal_info["lease_commence_date"])
                        if town_changed or lease_year_changed:
                            st.session_state.selected_town = postal_info["town"]
                            st.session_state.lease_commencement_year = int(postal_info["lease_commence_date"])
                            st.rerun()
            
            town_display_index = TOWNS.index(st.session_state.selected_town) if st.session_state.selected_town and st.session_state.selected_town in TOWNS else 0
            st.selectbox(
                "Town", TOWNS, index=town_display_index,
                disabled=True, key="town_disabled_display",
                help="Auto-filled from postal code if valid." if st.session_state.selected_town else "Will be auto-filled from valid postal code."
            )
        else:
            default_town_index = TOWNS.index("TAMPINES") if "TAMPINES" in TOWNS else 0
            current_selection_index = TOWNS.index(st.session_state.selected_town) if st.session_state.selected_town in TOWNS else default_town_index
            selected_town_manual = st.selectbox("Town", TOWNS, index=current_selection_index, key="town_active")
            if st.session_state.selected_town != selected_town_manual:
                 st.session_state.selected_town = selected_town_manual
            postal_code = None # Ensure postal code is None if not using postal toggle

        selected_flat_type = st.selectbox("Flat Type", FLAT_TYPES, index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0)
        selected_flat_model = st.selectbox("Flat Model", FLAT_MODELS, index=FLAT_MODELS.index("IMPROVED") if "IMPROVED" in FLAT_MODELS else 0)

    st.markdown("") 
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🔮 Predict Resale Price", type="primary", use_container_width=True)

    if predict_button:
        if use_postal and st.session_state.postal_validation_error:
            st.error("Please enter a valid postal code before making a prediction.")
        elif use_postal and not postal_code: 
             st.error("Postal code is enabled but not entered. Please enter a postal code.")
        elif not st.session_state.selected_town and not use_postal:
             st.error("Please select a town.")
        elif not st.session_state.selected_town and use_postal and not postal_code: 
            st.error("Please enter a postal code so town can be auto-filled.")
        elif xgb_model_loaded is None: 
            st.error("Model is not available. Prediction cannot be made. Please ensure 'model.bst' is in the correct location and check application logs.")
        else:
            with st.spinner("Analyzing market data..."):
                lease_commencement_year_value = st.session_state.lease_commencement_year
                lease_duration_at_sale = float(selected_year) - float(lease_commencement_year_value)
                calculated_remaining_lease_years = np.clip(99.0 - lease_duration_at_sale, 10.0, 99.0)
        
                input_values = {
                    'floor_area_sqm': float(floor_area),
                    'postal': float(postal_code) if postal_code and postal_code.isdigit() else 0.0, 
                    'remaining_lease_years': float(calculated_remaining_lease_years),
                    'storey_avg': float(storey),
                    'sale_year': float(selected_year),
                    'sale_month': float(MONTH_TO_NUM[selected_month])
                }
                final_input_for_model = {feat: 0.0 for feat in FEATURE_NAMES}
                for key, value in input_values.items():
                    if key in final_input_for_model: final_input_for_model[key] = value

                town_to_use = st.session_state.selected_town
                if not town_to_use: 
                    st.error("Town information is missing. Cannot proceed.")
                    st.stop()

                for prefix, selected_val, values_list in [
                    ("town_", town_to_use, TOWNS),
                    ("flat_type_", selected_flat_type, FLAT_TYPES),
                    ("flat_model_", selected_flat_model, FLAT_MODELS)
                ]:
                    feature_name = f"{prefix}{selected_val}"
                    if feature_name in final_input_for_model:
                        final_input_for_model[feature_name] = 1.0
                    else: st.warning(f"Selected value '{selected_val}' (feature: {feature_name}) not in model's FEATURE_NAMES. This may affect prediction accuracy.")
                
                try:
                    input_list = [final_input_for_model[feature] for feature in FEATURE_NAMES]
                    input_df = np.array(input_list).astype(np.float32).reshape(1, -1)
                    
                    if input_df.shape[1] != NUM_FEATURES:
                        st.error(f"Critical error: Input feature count ({input_df.shape[1]}) does not match model expectation ({NUM_FEATURES}).")
                        st.stop()

                    input_scaled = scaler.transform(input_df)
                    
                    dtest = xgb.DMatrix(input_scaled, feature_names=FEATURE_NAMES if xgb_model_loaded.num_features() == NUM_FEATURES else None) 
                    
                    prediction_array = xgb_model_loaded.predict(dtest)
                    prediction = float(prediction_array[0]) if prediction_array is not None and len(prediction_array) > 0 else None

                    if prediction is not None:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>Predicted Resale Price</h2>
                            <h1 style="font-size: 42px; color: {colors['success']};">S$ {prediction:,.2f}</h1>
                            <p>For {selected_flat_type} in {town_to_use} (Expected: {selected_month} {selected_year})</p>
                        </div>""", unsafe_allow_html=True)
                        with st.expander("View Property Details Summary"):
                            col_details1, col_details2 = st.columns(2)
                            with col_details1:
                                st.write("**Property Details:**")
                                if use_sqft: st.write(f"• Floor Area: {floor_area_sqft:.1f} sqft ({floor_area:.1f} sqm)")
                                else: st.write(f"• Floor Area: {floor_area:.1f} sqm ({sqm_to_sqft(floor_area):.1f} sqft)")
                                st.write(f"• Lease Commencement Year: {lease_commencement_year_value}")
                                st.write(f"• Remaining Lease (at sale): {calculated_remaining_lease_years:.1f} years")
                                st.write(f"• Storey (Average): {storey:.1f}")
                            with col_details2:
                                st.write("**Unit Details:**")
                                st.write(f"• Town: {town_to_use}")
                                if postal_code: st.write(f"• Postal Code: {postal_code}")
                                st.write(f"• Flat Type: {selected_flat_type}")
                                st.write(f"• Flat Model: {selected_flat_model}")
                    else: 
                        st.error("Prediction data could not be generated by the local model.")
                except xgb.core.XGBoostError as xgb_e:
                    st.error(f"XGBoost prediction error: {xgb_e}")
                  
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

elif st.session_state.active_page == PAGE_OPTIONS_LIST[1]: # "Transaction Map"
    st.markdown('<h2 class="sub-header">HDB Resale Transaction Map</h2>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">Click on any postal code marker to see the last 20 transactions for that location.</div>""", unsafe_allow_html=True)
    st.markdown('<div class="mobile-return-button-container">', unsafe_allow_html=True)
    if st.button("🏠", key="mobile_return_from_map_page"):
        st.session_state.active_page = PAGE_OPTIONS_LIST[0]
        st.session_state.menu_key_counter += 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="iframe-container">
        <iframe src="https://seehiong.github.io/app/hdb_resale_price_map_clickable.html" 
            style="width: 100%; height: 800px; border: none;" allow="geolocation"></iframe>
    </div>
    <script> 
    function checkMobileIframeHeight() {{
        var iframeContainer = document.querySelector('.iframe-container');
        if (iframeContainer) {{
            if (window.innerWidth <= 768) {{ iframeContainer.style.height = window.innerHeight * 0.8 + 'px'; }}
            else {{ iframeContainer.style.height = '800px'; }}
        }}
    }}
    window.addEventListener('load', checkMobileIframeHeight); 
    window.addEventListener('resize', checkMobileIframeHeight);
    </script>
    """, unsafe_allow_html=True)

elif st.session_state.active_page == PAGE_OPTIONS_LIST[2]: # "About"
    st.markdown(f"""
    ## About This Predictor
    This HDB resale price predictor uses a machine learning model to estimate the potential selling price of an HDB flat based on its characteristics and location.

    ### Model Information
    - **Model Type**: {MODEL_TYPE}
    - **Data Last Updated**: {DATA_INGESTION_DATE}
    - **Features Used**: Floor area, remaining lease (calculated from lease commencement year and sale date), storey, flat type, flat model, and location.

    ### How to Use
    1. Navigate to the "{PAGE_OPTIONS_LIST[0]}" page. Enter your property details, including the **Lease Commencement Year**.
    2. Select the expected sale date (year and month).
    3. Click "Predict Resale Price" to get an estimate.
    
    ### Postal Code Lookup
    - Toggle "Enter Postal Code Instead of Town" to automatically retrieve town and lease commencement information.
    - The system will validate the postal code against our database and auto-fill relevant details.

    ### Important Notes
    - This tool provides estimates only and should not be considered as financial advice.
    - Actual prices may vary based on market conditions and property specifics.
    - The model is trained on historical data and may not capture sudden market changes.
    """)