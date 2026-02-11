import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Laptop Price Intelligence",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%; background-color: #4CAF50; color: white;
        font-size: 18px; font-weight: bold; padding: 0.75rem;
        border-radius: 10px; border: none; margin-top: 1rem;
    }
    .stButton>button:hover {background-color: #45a049;}
    .prediction-box {
        padding: 2rem; border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; text-align: center; margin: 2rem 0;
    }
    .info-box {
        padding: 1rem; border-radius: 5px;
        background-color: #f0f2f6; margin: 1rem 0;
    }
    .product-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = 0
if 'prediction_inputs' not in st.session_state:
    st.session_state.prediction_inputs = {}

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Load dataset
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv('laptop_prices.csv')
    except:
        return None

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
model_name = model_data['model_name']
raw_data = load_raw_data()

# Get brand mappings
@st.cache_data
def get_brand_mappings():
    if raw_data is not None:
        brand_os = raw_data.groupby('Company')['OS'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        brand_type = raw_data.groupby('Company')['TypeName'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        brand_cpu = raw_data.groupby('Company')['CPU_company'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        brand_gpu = raw_data.groupby('Company')['GPU_company'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        return brand_os, brand_type, brand_cpu, brand_gpu
    else:
        companies = ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus']
        return ({c: ['Windows 10'] for c in companies}, 
                {c: ['Notebook'] for c in companies},
                {c: ['Intel'] for c in companies},
                {c: ['Intel'] for c in companies})

brand_os_map, brand_type_map, brand_cpu_map, brand_gpu_map = get_brand_mappings()
all_companies = sorted(brand_os_map.keys())

# Title
st.title("💻 Laptop Price Intelligence System")
st.markdown(f"""
    <div class="info-box">
        <h4>AI-Powered Price Predictor & Budget Finder</h4>
        <p>ML Model: <b>{model_name}</b> | R² Score: {model_data['test_r2_score']:.4f} | 
        RMSE: {model_data['test_rmse']:,.0f} Tsh</p>
    </div>
""", unsafe_allow_html=True)

# Mode selection
st.markdown("---")
mode = st.radio(
    "**Choose Your Mode:**",
    ["🔮 ML Price Predictor", "💰 Budget Finder"],
    horizontal=True,
    help="ML Predictor: Enter specs to predict price | Budget Finder: Enter budget to find laptops"
)
st.markdown("---")

# ============================================================================
# MODE 1: ML PRICE PREDICTOR
# ============================================================================
if mode == "🔮 ML Price Predictor":
    st.subheader("🔮 Machine Learning Price Predictor")
    st.markdown("*Enter laptop specifications to get an AI-powered price prediction*")
    
    # STEP 1: Brand
    with st.expander("📋 **STEP 1: Select Laptop Brand**", expanded=True):
        company = st.selectbox("Choose the laptop manufacturer", all_companies, key='company_select')

    # Get brand-specific options
    available_os = brand_os_map.get(company, ['Windows 10'])
    available_types = brand_type_map.get(company, ['Notebook'])
    available_cpu = brand_cpu_map.get(company, ['Intel'])
    available_gpu = brand_gpu_map.get(company, ['Intel'])

    # STEP 2: Specifications
    with st.expander("💻 **STEP 2: Configure Specifications**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            type_name = st.selectbox("Laptop Type", available_types, key='type_select')
            os = st.selectbox("Operating System", available_os, key='os_select')
            ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3, key='ram_select')
            primary_storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048], index=3, key='storage_select')
        with col2:
            cpu_company = st.selectbox("CPU Brand", available_cpu, key='cpu_select')
            gpu_company = st.selectbox("GPU Brand", available_gpu, key='gpu_select')
            cpu_freq = st.number_input("CPU Frequency (GHz)", 1.0, 4.0, 2.5, 0.1, key='cpu_freq_input')
            weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1, key='weight_input')

    # STEP 3: Display Features
    with st.expander("🖥️ **STEP 3: Display & Physical Features**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inches = st.number_input("Screen Size (inches)", 10.0, 18.0, 15.6, 0.1, key='inches_input')
            touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True, key='touch_select')
        with col2:
            ips_panel = st.radio("IPS Panel", ["No", "Yes"], horizontal=True, key='ips_select')
            retina_display = st.radio("Retina Display", ["No", "Yes"], horizontal=True, key='retina_select')

    # Prediction buttons
    st.markdown("---")
    col_predict, col_reset = st.columns([3, 1])
    with col_predict:
        predict_button = st.button("🔮 Predict Price", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()

    if predict_button:
        try:
            current_inputs = {
                'company': company, 'type_name': type_name, 'inches': inches,
                'ram': ram, 'os': os, 'weight': weight, 'touchscreen': touchscreen,
                'ips_panel': ips_panel, 'retina_display': retina_display,
                'cpu_company': cpu_company, 'cpu_freq': cpu_freq,
                'primary_storage': primary_storage, 'gpu_company': gpu_company
            }
            
            input_data = pd.DataFrame({
                'Company': [company], 'TypeName': [type_name], 'Inches': [inches],
                'Ram': [ram], 'OS': [os], 'Weight': [weight],
                'Touchscreen': [1 if touchscreen == "Yes" else 0],
                'IPSpanel': [1 if ips_panel == "Yes" else 0],
                'RetinaDisplay': [1 if retina_display == "Yes" else 0],
                'CPU_company': [cpu_company], 'CPU_freq': [cpu_freq],
                'PrimaryStorage': [primary_storage], 'GPU_company': [gpu_company]
            })
            
            for col in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']:
                le = label_encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col])
                except ValueError:
                    input_data[col] = 0
            
            input_data = input_data[feature_names]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.session_state.prediction_made = True
            st.session_state.predicted_price = prediction
            st.session_state.prediction_inputs = current_inputs
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Display prediction
    if st.session_state.prediction_made:
        prediction = st.session_state.predicted_price
        saved_inputs = st.session_state.prediction_inputs
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2>💰 Predicted Price</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:,.0f} Tsh</h1>
                <p style="font-size: 1.2rem;">≈ ${prediction/2500:.2f} USD</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        if raw_data is not None:
            brand_laptops = raw_data[raw_data['Company'] == saved_inputs['company']].copy()
            if len(brand_laptops) > 0:
                brand_laptops['price_diff'] = abs(brand_laptops['Price_Tsh'] - prediction)
                recommendations = brand_laptops.nsmallest(3, 'price_diff')
                
                st.markdown(f"### 🎯 Suggested Products from {saved_inputs['company']}")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    st.markdown(f"""
                        <div class="product-card">
                            <h4>{i}. {row['Company']} {row['Product']}</h4>
                            <p><b>💰 Price:</b> {row['Price_Tsh']:,.0f} Tsh | 
                            <b>💻 Type:</b> {row['TypeName']} | <b>🧠 RAM:</b> {row['Ram']} GB | 
                            <b>💾 Storage:</b> {row['PrimaryStorage']} GB</p>
                            <p><b>Difference:</b> {abs(row['Price_Tsh'] - prediction):,.0f} Tsh</p>
                        </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# MODE 2: BUDGET FINDER
# ============================================================================
elif mode == "💰 Budget Finder":
    st.subheader("💰 Find Laptops by Budget")
    st.markdown("*Enter your budget to see available laptops in that price range*")
    
    if raw_data is not None:
        # Budget input
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum Budget (Tsh)", 100000, 10000000, 1000000, 100000)
        with col2:
            max_price = st.number_input("Maximum Budget (Tsh)", 100000, 10000000, 3000000, 100000)
        
        # Filters
        st.markdown("#### Optional Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            brand_filter = st.multiselect("Brands", all_companies, default=[])
        with col2:
            type_filter = st.multiselect("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'], default=[])
        with col3:
            min_ram = st.selectbox("Minimum RAM (GB)", [0, 4, 8, 16, 32], index=0)
        
        # Search button
        if st.button("🔍 Search Laptops", use_container_width=True):
            results = raw_data[(raw_data['Price_Tsh'] >= min_price) & (raw_data['Price_Tsh'] <= max_price)]
            
            if brand_filter:
                results = results[results['Company'].isin(brand_filter)]
            if type_filter:
                results = results[results['TypeName'].isin(type_filter)]
            if min_ram > 0:
                results = results[results['Ram'] >= min_ram]
            
            results = results.sort_values('Price_Tsh')
            
            # Display results
            st.markdown(f"""
                <div class="info-box">
                    <h3>📊 Search Results: {len(results)} laptops found</h3>
                    <p>Budget: {min_price:,.0f} - {max_price:,.0f} Tsh</p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(results) > 0:
                # Show top 15 results
                for i, (_, laptop) in enumerate(results.head(15).iterrows(), 1):
                    st.markdown(f"""
                        <div class="product-card">
                            <h4>{i}. {laptop['Company']} {laptop['Product']}</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                                <div>
                                    <p><b>💰 Price:</b> {laptop['Price_Tsh']:,.0f} Tsh (≈ ${laptop['Price_Tsh']/2500:.0f})</p>
                                    <p><b>💻 Type:</b> {laptop['TypeName']}</p>
                                    <p><b>🖥️ OS:</b> {laptop['OS']}</p>
                                </div>
                                <div>
                                    <p><b>🧠 RAM:</b> {laptop['Ram']} GB | <b>💾 Storage:</b> {laptop['PrimaryStorage']} GB</p>
                                    <p><b>📺 Screen:</b> {laptop['Inches']}"</p>
                                    <p><b>⚙️ CPU:</b> {laptop['CPU_company']} {laptop['CPU_model']}</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                if len(results) > 15:
                    st.info(f"Showing top 15 of {len(results)} results. Refine filters to see more specific options.")
            else:
                st.warning("No laptops found in this price range with selected filters. Try adjusting your criteria.")
    else:
        st.error("Dataset not available. Please ensure laptop_prices.csv is in the app directory.")

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown(f"""
        ### 🎯 Two Modes Available:
        
        **🔮 ML Price Predictor**
        - Enter laptop specs
        - Get AI price prediction
        - See similar products
        
        **💰 Budget Finder**
        - Enter your budget
        - Find available laptops
        - Filter by preferences
        
        ---
        
        ### 📊 Model Info
        - Type: {model_name}
        - Accuracy: {model_data['test_r2_score']:.1%}
        - Features: {len(feature_names)}
        - Dataset: 1,275 laptops
        
        ---
        
        Made with ❤️ using ML & Streamlit
    """)
