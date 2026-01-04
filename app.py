# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
from datetime import datetime
from streamlit_option_menu import option_menu
import xgboost as xgb

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Calling st.rerun().*")

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="HDB Resale Price Predictor",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALISATION DES ÉTATS DE SESSION
# ============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

PAGE_OPTIONS_LIST = ["Make Prediction", "Transaction Map", "About"]
DATA_INGESTION_DATE = "16-07-2025"

if 'active_page' not in st.session_state:
    st.session_state.active_page = PAGE_OPTIONS_LIST[0]

if 'menu_key_counter' not in st.session_state:
    st.session_state.menu_key_counter = 0

# ============================================================================
# CONSTANTES ET CONFIGURATION
# ============================================================================
SCALER_PATH = "scaler.joblib"
POSTAL_DATA_PATH = "postal_data.json"
MODEL_LOCAL_PATH = "model.bst"
MODEL_TYPE = "XGBoost (Locally Loaded)"

FEATURE_NAMES = [
    'floor_area_sqm', 'storey_avg', 'sale_year', 'sale_month', 'remaining_lease_years',
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
    'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
    'flat_model_2-room', 'flat_model_3Gen', 'flat_model_Adjoined flat',
    'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved',
    'flat_model_Improved-Maisonette', 'flat_model_Model A',
    'flat_model_Model A-Maisonette', 'flat_model_Model A2',
    'flat_model_Multi Generation', 'flat_model_New Generation',
    'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft',
    'flat_model_Premium Maisonette', 'flat_model_Simplified',
    'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1',
    'flat_model_Type S2', 'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN',
    'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG',
    'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG',
    'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST',
    'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE',
    'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG',
    'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH',
    'town_WOODLANDS', 'town_YISHUN'
]

NUM_FEATURES = len(FEATURE_NAMES)

current_date = datetime.now()
FUTURE_YEARS = list(range(current_date.year - 10, current_date.year + 10))
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_TO_NUM = {month: i + 1 for i, month in enumerate(MONTHS)}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

def sqm_to_sqft(sqm):
    """Convert square meters to square feet"""
    return sqm * 10.7639

def sqft_to_sqm(sqft):
    """Convert square feet to square meters"""
    return sqft / 10.7639

def extract_categories(prefix, feature_list):
    """Extract category names from feature list based on prefix"""
    categories = [feature[len(prefix):] for feature in feature_list 
                 if feature.startswith(prefix)]
    return sorted(categories)

def validate_postal_code(postal_code, postal_data):
    """Validate postal code against database"""
    if not postal_code or not postal_code.isdigit() or len(postal_code) != 6:
        return None, "Please enter a valid 6-digit postal code."
    
    if not postal_data:
        return None, "Postal code database not loaded. Cannot validate."
    
    if postal_code not in postal_data:
        return None, f"Postal code {postal_code} not found in our database."
    
    return postal_data[postal_code][0], None

# ============================================================================
# CHARGEMENT DES RESSOURCES (CACHE)
# ============================================================================
@st.cache_resource
def load_scaler():
    """Load the scaler model"""
    try:
        if not os.path.exists(SCALER_PATH):
            st.error(f"Scaler file not found at {SCALER_PATH}")
            st.stop()
        return joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

@st.cache_resource
def load_postal_data():
    """Load postal code data"""
    try:
        if not os.path.exists(POSTAL_DATA_PATH):
            st.error(f"Postal data file not found at {POSTAL_DATA_PATH}")
            return {}
        with open(POSTAL_DATA_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading postal data: {e}")
        return {}

@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model"""
    try:
        if not os.path.exists(MODEL_LOCAL_PATH):
            st.error(f"Model file not found at {MODEL_LOCAL_PATH}")
            return None
        
        xgb_model = xgb.Booster()
        xgb_model.load_model(MODEL_LOCAL_PATH)
        
        # Validation des features
        try:
            num_model_features = xgb_model.num_features()
            if num_model_features != NUM_FEATURES:
                st.warning(f"Model feature count mismatch: {num_model_features} vs {NUM_FEATURES}")
        except Exception:
            pass
            
        return xgb_model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None

# ============================================================================
# INITIALISATION DES DONNÉES
# ============================================================================
scaler = load_scaler()
postal_data = load_postal_data()
model = load_xgboost_model()

FLAT_TYPES = extract_categories("flat_type_", FEATURE_NAMES)
FLAT_MODELS = extract_categories("flat_model_", FEATURE_NAMES)
TOWNS = extract_categories("town_", FEATURE_NAMES)

# Initialisation des états de session
if 'selected_town' not in st.session_state:
    st.session_state.selected_town = "TAMPINES" if "TAMPINES" in TOWNS else TOWNS[0]

if 'lease_commencement_year' not in st.session_state:
    st.session_state.lease_commencement_year = 1966

if 'postal_validation_error' not in st.session_state:
    st.session_state.postal_validation_error = None

# ============================================================================
# STYLISATION CSS
# ============================================================================
current_theme = st.session_state.theme
theme_colors = {
    'light': {
        'bg': '#FFFFFF', 'text': '#000000', 'primary': '#1E88E5',
        'secondary': '#0D47A1', 'success': '#43A047', 'info_bg': '#E3F2FD',
        'info_text': '#000000', 'info_border': '#1E88E5',
        'prediction_bg': '#E8F5E9', 'prediction_border': '#43A047',
        'sidebar_bg': '#F0F2F6', 'sidebar_text': '#000000'
    },
    'dark': {
        'bg': '#121212', 'text': '#FFFFFF', 'primary': '#90CAF9',
        'secondary': '#64B5F6', 'success': '#81C784', 'info_bg': '#0A1929',
        'info_text': '#E1E1E1', 'info_border': '#90CAF9',
        'prediction_bg': '#0A2018', 'prediction_border': '#81C784',
        'sidebar_bg': '#1E1E1E', 'sidebar_text': '#FFFFFF'
    }
}
colors = theme_colors[current_theme]

css = f"""
<style>
/* Styles de base */
.stApp {{
    max-width: 1200px;
    margin: 0 auto;
    background-color: {colors['bg']};
    color: {colors['text']};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {colors['sidebar_bg']} !important;
}}

[data-testid="stSidebar"] * {{
    color: {colors['sidebar_text']} !important;
}}

/* En-têtes */
.main-header {{
    color: {colors['primary']};
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 1rem;
}}

.sub-header {{
    color: {colors['secondary']};
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 1rem;
}}

/* Boîtes d'information */
.info-box {{
    background-color: {colors['info_bg']};
    color: {colors['info_text']};
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid {colors['info_border']};
    margin-bottom: 20px;
}}

/* Carte de prédiction */
.prediction-card {{
    background-color: {colors['prediction_bg']};
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin: 25px 0;
    border: 2px solid {colors['prediction_border']};
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}}

.prediction-price {{
    font-size: 48px;
    color: {colors['success']};
    font-weight: bold;
    margin: 15px 0;
}}

/* Inputs */
.stTextInput, .stNumberInput, .stSelectbox, .stDateInput {{
    margin-bottom: 15px;
}}

.stTextInput label, .stNumberInput label, 
.stSelectbox label, .stMultiSelect label {{
    color: {colors['text']} !important;
    font-weight: 500;
}}

/* Boutons */
.stButton > button {{
    background-color: {colors['primary']} !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    background-color: {colors['secondary']} !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}}

/* Menu d'options */
div[data-testid="stOptionMenu"] button {{
    color: {colors['text']} !important;
    border-radius: 6px !important;
    margin: 0 5px !important;
    padding: 10px 20px !important;
}}

div[data-testid="stOptionMenu"] button[aria-selected="true"] {{
    background-color: {colors['primary']} !important;
    color: white !important;
}}

/* Adaptatif mobile */
@media (max-width: 768px) {{
    .stApp {{ padding: 10px; }}
    .main-header {{ font-size: 28px !important; }}
    .sub-header {{ font-size: 20px !important; }}
    .prediction-price {{ font-size: 36px !important; }}
    
    div[data-testid="stOptionMenu"] button {{
        font-size: 14px !important;
        padding: 8px 12px !important;
        margin: 2px !important;
    }}
}}

/* Séparateurs */
hr {{
    border-color: {colors['primary']}20;
    margin: 25px 0;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Toggle theme
    theme_label = "🌙 Dark Mode" if st.session_state.theme == 'light' else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.markdown("---")
    
    # Navigation rapide
    st.markdown("### 🧭 Navigation")
    for page in PAGE_OPTIONS_LIST:
        if st.button(f"📌 {page}", use_container_width=True, 
                    type="primary" if page == st.session_state.active_page else "secondary"):
            st.session_state.active_page = page
            st.session_state.menu_key_counter += 1
            st.rerun()
    
    st.markdown("---")
    
    # Informations
    st.markdown("### ℹ️ Information")
    st.markdown(f"**Model:** {MODEL_TYPE}")
    st.markdown(f"**Data Updated:** {DATA_INGESTION_DATE}")
    st.markdown(f"**Theme:** {current_theme.title()}")

# ============================================================================
# EN-TÊTE PRINCIPAL
# ============================================================================
st.markdown('<h1 class="main-header">🏙️ HDB Resale Price Predictor</h1>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
        <span style="font-size: 24px;">🤖</span>
        <strong style="font-size: 16px;">AI-Powered Price Estimation</strong>
    </div>
    <p>This cutting-edge predictor uses XGBoost machine learning trained on extensive historical data to estimate HDB resale prices with high accuracy.</p>
    <p style="margin-top: 10px; font-size: 14px; color: {colors['info_text']}80;">
        Model: {MODEL_TYPE} | Data Last Updated: {DATA_INGESTION_DATE}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MENU DE NAVIGATION
# ============================================================================
icons_list = ['📊', '🗺️', 'ℹ️']
default_nav_index = PAGE_OPTIONS_LIST.index(st.session_state.active_page)
current_menu_key = f"main_nav_menu_{st.session_state.menu_key_counter}"

selected_page_from_menu = option_menu(
    menu_title=None,
    options=PAGE_OPTIONS_LIST,
    icons=icons_list,
    default_index=default_nav_index,
    orientation="horizontal",
    key=current_menu_key,
    styles={
        "container": {"padding": "0!important", "margin": "20px 0"},
        "nav-link": {"font-size": "16px", "margin": "0 5px"},
        "nav-link-selected": {"background-color": colors['primary']},
    }
)

if selected_page_from_menu != st.session_state.active_page:
    st.session_state.active_page = selected_page_from_menu
    st.rerun()

# ============================================================================
# PAGE: MAKE PREDICTION
# ============================================================================
if st.session_state.active_page == PAGE_OPTIONS_LIST[0]:
    
    # Section 1: Date de vente
    st.markdown('<h2 class="sub-header">📅 Expected Sale Date</h2>', unsafe_allow_html=True)
    col_year, col_month = st.columns(2)
    
    with col_year:
        default_year_idx = FUTURE_YEARS.index(current_date.year) if current_date.year in FUTURE_YEARS else len(FUTURE_YEARS) // 2
        selected_year = st.selectbox(
            "Sale Year", 
            FUTURE_YEARS, 
            index=default_year_idx,
            help="Select the year you plan to sell your property"
        )
    
    with col_month:
        selected_month = st.selectbox(
            "Sale Month", 
            MONTHS, 
            index=current_date.month - 1,
            help="Select the month you plan to sell your property"
        )
    
    st.markdown("---")
    
    # Section 2: Détails de la propriété
    col_property, col_location = st.columns(2)
    
    with col_property:
        st.markdown('<h2 class="sub-header">🏠 Property Details</h2>', unsafe_allow_html=True)
        
        # Toggle pour unités de surface
        use_sqft = st.toggle(
            "Use Square Feet (sqft)", 
            value=False,
            help="Toggle between square meters and square feet"
        )
        
        # Input de surface
        if use_sqft:
            floor_area_sqft = st.number_input(
                "Floor Area (sqft)", 
                min_value=215.0, 
                max_value=3230.0, 
                value=969.0, 
                step=10.0,
                format="%.1f",
                help="Enter the floor area in square feet"
            )
            floor_area = sqft_to_sqm(floor_area_sqft)
            st.caption(f"≈ {floor_area:.1f} sqm")
        else:
            floor_area = st.number_input(
                "Floor Area (sqm)", 
                min_value=20.0, 
                max_value=300.0, 
                value=90.0, 
                step=1.0,
                format="%.1f",
                help="Enter the floor area in square meters"
            )
            st.caption(f"≈ {sqm_to_sqft(floor_area):.1f} sqft")
        
        # Année de commencement du bail
        lease_commencement_year = st.number_input(
            "Lease Commencement Year", 
            min_value=1966, 
            max_value=current_date.year,
            value=st.session_state.lease_commencement_year, 
            step=1,
            help="Enter the year the flat's 99-year lease started",
            key="lease_year_input"
        )
        
        # Calcul du bail restant
        remaining_years = 99 - (selected_year - lease_commencement_year)
        remaining_years = max(10, min(99, remaining_years))
        st.info(f"Remaining lease at sale: **{remaining_years:.1f} years**")
        
        # Étage moyen
        storey = st.number_input(
            "Storey (Average)", 
            min_value=1.0, 
            max_value=50.0, 
            value=10.0, 
            step=1.0,
            format="%.1f",
            help="Enter the average storey level"
        )
    
    with col_location:
        st.markdown('<h2 class="sub-header">📍 Location & Type</h2>', unsafe_allow_html=True)
        
        # Toggle pour code postal
        use_postal = st.toggle(
            "Enter Postal Code Instead of Town", 
            value=False,
            help="Enable to auto-fill town and lease details from postal code"
        )
        
        postal_error_container = st.empty()
        
        if use_postal:
            # Input code postal
            postal_code = st.text_input(
                "Postal Code", 
                placeholder="e.g., 760123",
                max_chars=6,
                help="Enter a 6-digit Singapore postal code"
            )
            
            # Validation du code postal
            if postal_code:
                postal_info, error_msg = validate_postal_code(postal_code, postal_data)
                
                if error_msg:
                    postal_error_container.error(error_msg)
                    st.session_state.postal_validation_error = error_msg
                    st.session_state.selected_town = None
                else:
                    postal_error_container.success(f"✅ Validated: {postal_code}")
                    st.session_state.postal_validation_error = None
                    
                    if postal_info:
                        # Mise à jour des informations
                        town_changed = st.session_state.selected_town != postal_info["town"]
                        lease_changed = st.session_state.lease_commencement_year != int(postal_info["lease_commence_date"])
                        
                        if town_changed or lease_changed:
                            st.session_state.selected_town = postal_info["town"]
                            st.session_state.lease_commencement_year = int(postal_info["lease_commence_date"])
                            st.rerun()
            
            # Affichage de la ville (lecture seule)
            town_display_index = TOWNS.index(st.session_state.selected_town) if st.session_state.selected_town in TOWNS else 0
            st.selectbox(
                "Town", 
                TOWNS, 
                index=town_display_index,
                disabled=True,
                help="Auto-filled from postal code"
            )
        else:
            # Sélection manuelle de la ville
            default_town_index = TOWNS.index("TAMPINES") if "TAMPINES" in TOWNS else 0
            current_index = TOWNS.index(st.session_state.selected_town) if st.session_state.selected_town in TOWNS else default_town_index
            
            selected_town = st.selectbox(
                "Town", 
                TOWNS, 
                index=current_index,
                help="Select the town where the property is located"
            )
            
            if st.session_state.selected_town != selected_town:
                st.session_state.selected_town = selected_town
            
            postal_code = None
        
        # Type d'appartement
        selected_flat_type = st.selectbox(
            "Flat Type", 
            FLAT_TYPES, 
            index=FLAT_TYPES.index("4 ROOM") if "4 ROOM" in FLAT_TYPES else 0,
            help="Select the flat type"
        )
        
        # Modèle d'appartement
        selected_flat_model = st.selectbox(
            "Flat Model", 
            FLAT_MODELS, 
            index=FLAT_MODELS.index("Model A") if "Model A" in FLAT_MODELS else 0,
            help="Select the flat model"
        )
    
    st.markdown("---")
    
    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Resale Price", 
            type="primary", 
            use_container_width=True,
            disabled=(model is None)
        )
    
    # ============================================================================
    # TRAITEMENT DE LA PRÉDICTION
    # ============================================================================
    if predict_button:
        # Validation des entrées
        validation_errors = []
        
        if use_postal:
            if st.session_state.postal_validation_error:
                validation_errors.append("Please enter a valid postal code.")
            elif not postal_code:
                validation_errors.append("Postal code is enabled but not entered.")
        
        if not st.session_state.selected_town:
            validation_errors.append("Please select a town.")
        
        if model is None:
            validation_errors.append("Model is not available. Please check the model file.")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            with st.spinner("🔍 Analyzing market data and making prediction..."):
                try:
                    # Préparation des données
                    lease_duration = float(selected_year) - float(lease_commencement_year)
                    remaining_lease = np.clip(99.0 - lease_duration, 10.0, 99.0)
                    
                    # Construction du vecteur d'entrée
                    input_features = {feat: 0.0 for feat in FEATURE_NAMES}
                    
                    # Features numériques
                    numeric_features = {
                        'floor_area_sqm': float(floor_area),
                        'storey_avg': float(storey),
                        'sale_year': float(selected_year),
                        'sale_month': float(MONTH_TO_NUM[selected_month]),
                        'remaining_lease_years': float(remaining_lease)
                    }
                    
                    for key, value in numeric_features.items():
                        if key in input_features:
                            input_features[key] = value
                    
                    # Features catégorielles
                    categorical_mappings = [
                        ("town_", st.session_state.selected_town),
                        ("flat_type_", selected_flat_type),
                        ("flat_model_", selected_flat_model)
                    ]
                    
                    for prefix, value in categorical_mappings:
                        feature_name = f"{prefix}{value}"
                        if feature_name in input_features:
                            input_features[feature_name] = 1.0
                        else:
                            st.warning(f"Feature '{feature_name}' not in model. This may affect accuracy.")
                    
                    # Conversion en array
                    input_array = np.array([input_features[feat] for feat in FEATURE_NAMES], 
                                          dtype=np.float32).reshape(1, -1)
                    
                    if input_array.shape[1] != NUM_FEATURES:
                        st.error(f"Feature count mismatch: {input_array.shape[1]} vs {NUM_FEATURES}")
                        st.stop()
                    
                    # Normalisation et prédiction
                    input_scaled = scaler.transform(input_array)
                    dmatrix = xgb.DMatrix(input_scaled, feature_names=FEATURE_NAMES)
                    prediction = float(model.predict(dmatrix)[0])
                    
                    # ============================================================================
                    # AFFICHAGE DES RÉSULTATS
                    # ============================================================================
                    st.markdown("---")
                    
                    # Carte de prédiction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>📊 Predicted Resale Price</h2>
                        <div class="prediction-price">S$ {prediction:,.0f}</div>
                        <p style="font-size: 16px; color: {colors['text']}80;">
                            For a {selected_flat_type.lower()} in {st.session_state.selected_town}<br>
                            Expected sale: {selected_month} {selected_year}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Détails expansibles
                    with st.expander("📋 View Property Details", expanded=False):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("**Property Specifications**")
                            st.markdown(f"""
                            - **Floor Area:** {floor_area:.1f} sqm ({sqm_to_sqft(floor_area):.1f} sqft)
                            - **Storey Level:** {storey:.1f}
                            - **Lease Commencement:** {lease_commencement_year}
                            - **Remaining Lease:** {remaining_lease:.1f} years
                            """)
                        
                        with detail_col2:
                            st.markdown("**Location & Type**")
                            details = f"""
                            - **Town:** {st.session_state.selected_town}
                            - **Flat Type:** {selected_flat_type}
                            - **Flat Model:** {selected_flat_model}
                            """
                            if postal_code:
                                details += f"- **Postal Code:** {postal_code}\n"
                            st.markdown(details)
                    
                    # Note informative
                    st.info("""
                    **Note:** This prediction is based on historical data and machine learning algorithms. 
                    Actual resale prices may vary due to market conditions, property condition, 
                    negotiation factors, and other variables not captured by the model.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {str(e)}")
                    st.error("Please check your inputs and try again.")

# ============================================================================
# PAGE: TRANSACTION MAP
# ============================================================================
elif st.session_state.active_page == PAGE_OPTIONS_LIST[1]:
    st.markdown('<h2 class="sub-header">🗺️ HDB Resale Transaction Map</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p><strong>Interactive Map Features:</strong></p>
        <ul style="margin-bottom: 0;">
            <li>Click on any postal code marker to view recent transactions</li>
            <li>Zoom in/out to explore different regions</li>
            <li>View last 20 transactions for each location</li>
            <li>Data updated regularly from official sources</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Conteneur pour la carte
    map_html = """
    <div style="width: 100%; height: 700px; border-radius: 10px; overflow: hidden; margin-top: 20px;">
        <iframe 
            src="https://seehiong.github.io/app/hdb_resale_price_map_clickable.html" 
            style="width: 100%; height: 100%; border: none;"
            title="HDB Resale Transaction Map"
            allow="geolocation"
            loading="lazy">
        </iframe>
    </div>
    """
    
    st.markdown(map_html, unsafe_allow_html=True)
    
    # Instructions mobiles
    st.markdown("""
    <div style="margin-top: 20px; padding: 15px; background-color: %s; border-radius: 8px;">
        <p style="margin: 0; font-size: 14px;"><strong>📱 Mobile Users:</strong> 
        Use two fingers to pinch and zoom on the map. Tap markers to see transaction details.</p>
    </div>
    """ % colors['info_bg'], unsafe_allow_html=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
else:
    st.markdown('<h2 class="sub-header">ℹ️ About This Predictor</h2>', unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
        ### 🎯 Purpose
        This HDB Resale Price Predictor leverages advanced machine learning 
        to estimate property values based on historical transaction data, 
        helping homeowners and potential buyers make informed decisions.
        
        ### 🤖 Technology
        - **Algorithm:** XGBoost (Extreme Gradient Boosting)
        - **Training Data:** Extensive historical HDB resale records
        - **Features:** 58 key property characteristics
        - **Accuracy:** Continuously improved with new data
        
        ### 📊 Data Sources
        - Official HDB resale transaction records
        - Singapore postal code database
        - Property market trends and indices
        """)
    
    with col_about2:
        st.markdown("""
        ### 🚀 How to Use
        1. **Enter Property Details** - Fill in floor area, lease year, storey
        2. **Select Location** - Choose town or enter postal code
        3. **Set Sale Date** - Specify expected sale month and year
        4. **Get Prediction** - Click predict for AI-powered estimate
        
        ### 🔍 Key Features
        - Postal code auto-completion
        - Lease duration calculation
        - Mobile-responsive design
        - Dark/Light mode toggle
        - Interactive transaction map
        
        ### ⚠️ Important Notes
        - Estimates are for reference only
        - Not financial or investment advice
        - Market conditions may affect actual prices
        - Regular model updates for accuracy
        """)
    
    st.markdown("---")
    
    # Section technique
    with st.expander("📈 Technical Details", expanded=False):
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            ### Model Specifications
            - **Type:** Regression (XGBoost)
            - **Features:** 58 engineered features
            - **Training Period:** 2015-2024
            - **Update Frequency:** Quarterly
            - **Performance:** RMSE ~$40,000
            
            ### Features Included
            - Location (Town, Postal Sector)
            - Property Characteristics
            - Temporal Factors
            - Lease Information
            - Market Trends
            """)
        
        with tech_col2:
            st.markdown("""
            ### Data Processing
            - **Cleaning:** Outlier removal, missing value imputation
            - **Encoding:** One-hot for categorical variables
            - **Scaling:** StandardScaler for numerical features
            - **Validation:** 80/20 train-test split
            - **Cross-validation:** 5-fold for hyperparameter tuning
            
            ### Limitations
            - Cannot predict market shocks
            - Limited by historical patterns
            - Excludes interior condition
            - Neighborhood effects simplified
            """)
    
    # Contact/Info
    st.markdown(f"""
    <div style="margin-top: 30px; padding: 20px; background-color: {colors['info_bg']}; 
                border-radius: 10px; text-align: center;">
        <p style="margin: 0;">
            <strong>Data Last Updated:</strong> {DATA_INGESTION_DATE} | 
            <strong>Model Version:</strong> XGBoost 2.0 | 
            <strong>Contact:</strong> support@hdbpredictor.sg
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PIED DE PAGE
# ============================================================================
st.markdown("""
<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid %s; 
            text-align: center; font-size: 12px; color: %s80;">
    <p>© 2024 HDB Resale Price Predictor. This tool is for informational purposes only. 
    Prices are estimates based on historical data and machine learning models.</p>
    <p>Always consult with real estate professionals for accurate valuation and advice.</p>
</div>
""" % (colors['primary'], colors['text']), unsafe_allow_html=True)
