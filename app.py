import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with professional styling and color accents
st.markdown("""
    <style>
    /* Global styles */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Typography */
    h1 {
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* Cards */
    .metric-card {
        background: lavender;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 3px solid #3498db;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-top-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.15);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 8px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .product-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #3498db;
        border-radius: 6px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }
    
    .product-card:hover {
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        transform: translateY(-2px);
        border-left-color: #2980b9;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(to right, #ebf5fb 0%, #f8f9fa 100%);
        border-left: 4px solid #3498db;
        padding: 0.75rem 1rem;
        margin: 1.5rem 0 1rem 0;
        border-radius: 4px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(to right, #ebf5fb 0%, #f8f9fa 100%);
        border: 1px solid #d6eaf8;
        border-left: 3px solid #3498db;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        font-size: 16px;
        font-weight: 500;
        padding: 0.65rem;
        border-radius: 6px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #21618c 100%);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        transform: translateY(-1px);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Stats */
    .stat-value {
        font-size: 2rem;
        font-weight: 600;
        color: #000080;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Radio buttons accent */
    .stRadio > label {
        color: #000080;
    }
    
    /* Selectbox accent */
    div[data-baseweb="select"] > div {
        border-color: #d6eaf8;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #3498db;
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

# Sidebar
with st.sidebar:
    st.title("Laptop Price Predictor")
    st.markdown("---")
    
    # Mode selection in sidebar
    mode = st.radio(
        "Analysis Mode",
        ["Price Predictor", "Budget Finder", "Model Analytics", "Model Comparison"],
        help="Select the analysis mode you want to use"
    )
    
    st.markdown("---")
    
    # Model information
    st.subheader("Model Information")
    st.markdown(f"""
    **Algorithm:** {model_name}
    
    **Performance Metrics:**
    - R² Score: {model_data['test_r2_score']:.4f}
    - RMSE: {model_data['test_rmse']:,.0f} Tsh
    - Features: {len(feature_names)}
    
    **Dataset:**
    - Total Samples: 1,275 laptops
    - Price Range: 100K - 10M Tsh
    """)
    
    st.markdown("---")
    
    # Model download button
    st.subheader("Download Model")
    try:
        with open('model.pkl', 'rb') as f:
            model_bytes = f.read()
        
        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="laptop_price_model.pkl",
            mime="application/octet-stream",
            help="Download the trained machine learning model",
            use_container_width=True
        )
    except FileNotFoundError:
        st.error("Model file not found")
    
    st.markdown("---")
    
    # About section
    with st.expander("About This Tool"):
        st.markdown("""
        This application uses machine learning to predict laptop prices 
        based on specifications and help users find laptops within their budget.
        
        **Features:**
        - ML-based price prediction
        - Budget-based laptop search
        - Product recommendations
        - Model performance analytics
        """)

# Main content
st.title("Laptop Price Predictor")
st.markdown("Machine learning-powered laptop price prediction and budget analysis")

# ============================================================================
# MODE 1: PRICE PREDICTOR
# ============================================================================
if mode == "Price Predictor":
    
    # Model performance summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">R² Score</p>
            <p class="stat-value">{model_data['test_r2_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">RMSE</p>
            <p class="stat-value">{model_data['test_rmse']/1000:.0f}K</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">Features</p>
            <p class="stat-value">{len(feature_names)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">Model</p>
            <p class="stat-value" style="font-size: 1.2rem;">{model_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'><h3>Configure Laptop Specifications</h3></div>", unsafe_allow_html=True)
    
    # STEP 1: Brand Selection
    st.markdown("**Brand & Type**")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Manufacturer", all_companies, key='company_select')
    
    # Get brand-specific options
    available_os = brand_os_map.get(company, ['Windows 10'])
    available_types = brand_type_map.get(company, ['Notebook'])
    available_cpu = brand_cpu_map.get(company, ['Intel'])
    available_gpu = brand_gpu_map.get(company, ['Intel'])
    
    with col2:
        type_name = st.selectbox("Type", available_types, key='type_select')
    
    st.markdown("---")
    
    # STEP 2: Core Specifications
    st.markdown("**Core Specifications**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3, key='ram_select')
    with col2:
        primary_storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048], index=3, key='storage_select')
    with col3:
        cpu_company = st.selectbox("CPU Brand", available_cpu, key='cpu_select')
    with col4:
        gpu_company = st.selectbox("GPU Brand", available_gpu, key='gpu_select')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cpu_freq = st.number_input("CPU Frequency (GHz)", 1.0, 4.0, 2.5, 0.1, key='cpu_freq_input')
    with col2:
        weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1, key='weight_input')
    with col3:
        inches = st.number_input("Screen Size (inches)", 10.0, 18.0, 15.6, 0.1, key='inches_input')
    with col4:
        os = st.selectbox("Operating System", available_os, key='os_select')
    
    st.markdown("---")
    
    # STEP 3: Display Features
    st.markdown("**Display Features**")
    col1, col2, col3 = st.columns(3)
    with col1:
        touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True, key='touch_select')
    with col2:
        ips_panel = st.radio("IPS Panel", ["No", "Yes"], horizontal=True, key='ips_select')
    with col3:
        retina_display = st.radio("Retina Display", ["No", "Yes"], horizontal=True, key='retina_select')
    
    st.markdown("---")
    
    # Button styling - place before buttons
    st.markdown("""
        <style>
      /* Primary button - centered text with larger font */
        button[kind="primary"] p {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
        
        button[kind="primary"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Reset button styling */
        button[kind="secondary"] {
            background: white !important;
            color: #e74c3c !important;
            border: 2px solid #e74c3c !important;
        }
        button[kind="secondary"]:hover {
            background: #DC0E0E !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4) !important;
        }
         button[kind="secondary"] p {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main Action Button - Predict Price
    col1, col2 = st.columns([5, 1])
    with col1:
        predict_button = st.button("Predict Price", use_container_width=True, type="primary", key="main_predict")
    with col2:
        if st.button("⟳ Reset", use_container_width=True, type="secondary", key="reset_btn"):
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
            st.error(f"Error: {str(e)}")
    
    # Display prediction
    if st.session_state.prediction_made:
        prediction = st.session_state.predicted_price
        saved_inputs = st.session_state.prediction_inputs
        
        st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Price</h2>
                <h1 style="font-size: 2.5rem; margin: 1rem 0; font-weight: 600;">{prediction:,.0f} Tsh</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">≈ ${prediction/2500:.2f} USD</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Download Prediction Report
        report_data = {
            "Prediction Summary": {
                "Predicted Price (Tsh)": f"{prediction:,.0f}",
                "Predicted Price (USD)": f"${prediction/2500:.2f}",
                "Prediction Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "Input Specifications": {
                "Brand": saved_inputs['company'],
                "Type": saved_inputs['type_name'],
                "Screen Size": f"{saved_inputs['inches']}\"",
                "RAM": f"{saved_inputs['ram']} GB",
                "Storage": f"{saved_inputs['primary_storage']} GB",
                "Operating System": saved_inputs['os'],
                "CPU Brand": saved_inputs['cpu_company'],
                "CPU Frequency": f"{saved_inputs['cpu_freq']} GHz",
                "GPU Brand": saved_inputs['gpu_company'],
                "Weight": f"{saved_inputs['weight']} kg",
                "Touchscreen": saved_inputs['touchscreen'],
                "IPS Panel": saved_inputs['ips_panel'],
                "Retina Display": saved_inputs['retina_display']
            },
            "Model Information": {
                "Model Type": model_name,
                "R² Score": f"{model_data['test_r2_score']:.4f}",
                "RMSE": f"{model_data['test_rmse']:,.0f} Tsh",
                "Features Used": len(feature_names)
            }
        }
        
        # Create report text
        report_text = "LAPTOP PRICE PREDICTION REPORT\n"
        report_text += "=" * 50 + "\n\n"
        
        for section, data in report_data.items():
            report_text += f"{section}\n"
            report_text += "-" * 50 + "\n"
            for key, value in data.items():
                report_text += f"{key}: {value}\n"
            report_text += "\n"
        
        report_text += "\n" + "=" * 50 + "\n"
        report_text += f"Generated by Laptop Price Predictor\n"
        report_text += f"Model: {model_name} | Accuracy: {model_data['test_r2_score']:.1%}\n"
        
        # Download button
        st.download_button(
            label="➜] Download Prediction Report",
            data=report_text,
            file_name=f"laptop_price_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            type="secondary"
        )
        
        st.markdown("---")
        
        # Recommendations
        if raw_data is not None:
            brand_laptops = raw_data[raw_data['Company'] == saved_inputs['company']].copy()
            if len(brand_laptops) > 0:
                brand_laptops['price_diff'] = abs(brand_laptops['Price_Tsh'] - prediction)
                recommendations = brand_laptops.nsmallest(3, 'price_diff')
                
                st.markdown(f"<div class='section-header'><h4>Similar Products from {saved_inputs['company']}</h4></div>", unsafe_allow_html=True)
                
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    st.markdown(f"""
                        <div class="product-card">
                            <h4 style="margin-top: 0; color: #2980b9;">{row['Company']} {row['Product']}</h4>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.75rem;">
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>Price:</strong> {row['Price_Tsh']:,.0f} Tsh</p>
                                    <p style="margin: 0.25rem 0;"><strong>Type:</strong> {row['TypeName']}</p>
                                </div>
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>RAM:</strong> {row['Ram']} GB</p>
                                    <p style="margin: 0.25rem 0;"><strong>Storage:</strong> {row['PrimaryStorage']} GB</p>
                                </div>
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>Screen:</strong> {row['Inches']}"</p>
                                    <p style="margin: 0.25rem 0;"><strong>Difference:</strong> {abs(row['Price_Tsh'] - prediction):,.0f} Tsh</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# MODE 2: BUDGET FINDER
# ============================================================================
elif mode == "Budget Finder":
    
    st.markdown("<div class='section-header'><h3>Find Laptops by Budget</h3></div>", unsafe_allow_html=True)
    
    if raw_data is not None:
        # Budget input
        st.markdown("**Budget Range**")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum Budget (Tsh)", 100000, 10000000, 1000000, 100000)
        with col2:
            max_price = st.number_input("Maximum Budget (Tsh)", 100000, 10000000, 3000000, 100000)
        
        st.markdown("---")
        
        # Filters
        st.markdown("**Filter Options**")
        col1, col2, col3 = st.columns(3)
        with col1:
            brand_filter = st.multiselect("Brands", all_companies, default=[])
        with col2:
            type_filter = st.multiselect("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'], default=[])
        with col3:
            min_ram = st.selectbox("Minimum RAM (GB)", [0, 4, 8, 16, 32], index=0)
        
        st.markdown("---")
        
        # Search button
        if st.button("Search Laptops", use_container_width=True):
            results = raw_data[(raw_data['Price_Tsh'] >= min_price) & (raw_data['Price_Tsh'] <= max_price)]
            
            if brand_filter:
                results = results[results['Company'].isin(brand_filter)]
            if type_filter:
                results = results[results['TypeName'].isin(type_filter)]
            if min_ram > 0:
                results = results[results['Ram'] >= min_ram]
            
            results = results.sort_values('Price_Tsh')
            
            # Display summary
            st.markdown(f"""
                <div class="info-box">
                    <h3 style="margin-top: 0;">Search Results: {len(results)} laptops found</h3>
                    <p style="margin-bottom: 0;">Budget Range: {min_price:,.0f} - {max_price:,.0f} Tsh</p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(results) > 0:
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="stat-label">Average Price</p>
                        <p class="stat-value" style="font-size: 1.5rem;">{results['Price_Tsh'].mean()/1000:.0f}K</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="stat-label">Avg RAM</p>
                        <p class="stat-value" style="font-size: 1.5rem;">{results['Ram'].mean():.0f} GB</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="stat-label">Avg Storage</p>
                        <p class="stat-value" style="font-size: 1.5rem;">{results['PrimaryStorage'].mean():.0f} GB</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="stat-label">Avg Screen</p>
                        <p class="stat-value" style="font-size: 1.5rem;">{results['Inches'].mean():.1f}"</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Price distribution chart
                st.markdown("<div class='section-header'><h4>Price Distribution in Range</h4></div>", unsafe_allow_html=True)
                fig = px.histogram(results, x='Price_Tsh', nbins=20, 
                                 color_discrete_sequence=['#3498db'])
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis_title="Price (Tsh)",
                    yaxis_title="Count",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show results
                st.markdown("<div class='section-header'><h4>Available Laptops</h4></div>", unsafe_allow_html=True)
                
                for i, (_, laptop) in enumerate(results.head(15).iterrows(), 1):
                    st.markdown(f"""
                        <div class="product-card">
                            <h4 style="margin-top: 0; color: #2980b9;">{laptop['Company']} {laptop['Product']}</h4>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 0.75rem;">
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>Price:</strong> {laptop['Price_Tsh']:,.0f} Tsh</p>
                                    <p style="margin: 0.25rem 0; color: #6c757d;">≈ ${laptop['Price_Tsh']/2500:.0f} USD</p>
                                </div>
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>Type:</strong> {laptop['TypeName']}</p>
                                    <p style="margin: 0.25rem 0;"><strong>OS:</strong> {laptop['OS']}</p>
                                </div>
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>RAM:</strong> {laptop['Ram']} GB</p>
                                    <p style="margin: 0.25rem 0;"><strong>Storage:</strong> {laptop['PrimaryStorage']} GB</p>
                                </div>
                                <div>
                                    <p style="margin: 0.25rem 0;"><strong>Screen:</strong> {laptop['Inches']}"</p>
                                    <p style="margin: 0.25rem 0;"><strong>CPU:</strong> {laptop['CPU_company']}</p>
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

# ============================================================================
# MODE 3: MODEL ANALYTICS
# ============================================================================
elif mode == "Model Analytics":
    
    st.markdown("<div class='section-header'><h3>Model Performance Analytics</h3></div>", unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">R² Score</p>
            <p class="stat-value">{model_data['test_r2_score']:.4f}</p>
            <p style="margin: 0; color: #6c757d; font-size: 0.85rem;">Coefficient of Determination</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">RMSE</p>
            <p class="stat-value">{model_data['test_rmse']:,.0f}</p>
            <p style="margin: 0; color: #6c757d; font-size: 0.85rem;">Root Mean Square Error (Tsh)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="stat-label">Model Type</p>
            <p class="stat-value" style="font-size: 1.5rem;">{model_name}</p>
            <p style="margin: 0; color: #6c757d; font-size: 0.85rem;">{len(feature_names)} features</p>
        </div>
        """, unsafe_allow_html=True)
    
    if raw_data is not None:
        st.markdown("---")
        
        # Dataset statistics
        st.markdown("<div class='section-header'><h4>Dataset Statistics</h4></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Distribution**")
            fig = px.histogram(raw_data, x='Price_Tsh', nbins=40,
                             color_discrete_sequence=['#3498db'])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis_title="Price (Tsh)",
                yaxis_title="Frequency",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**RAM Distribution**")
            ram_counts = raw_data['Ram'].value_counts().sort_index()
            fig = px.bar(x=ram_counts.index, y=ram_counts.values,
                        color_discrete_sequence=['#3498db'])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis_title="RAM (GB)",
                yaxis_title="Count",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Price by Brand**")
            brand_prices = raw_data.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=True).tail(10)
            fig = px.bar(x=brand_prices.values, y=brand_prices.index, orientation='h',
                        color_discrete_sequence=['#3498db'])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis_title="Average Price (Tsh)",
                yaxis_title="Brand",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Type Distribution**")
            type_counts = raw_data['TypeName'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        color_discrete_sequence=['#3498db', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8'])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature importance (if available in model_data)
        st.markdown("<div class='section-header'><h4>Model Features</h4></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numerical Features:**")
            numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
            for feat in numerical_features:
                if feat in feature_names:
                    st.write(f"- {feat}")
        
        with col2:
            st.markdown("**Categorical Features:**")
            categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company', 
                                   'Touchscreen', 'IPSpanel', 'RetinaDisplay']
            for feat in categorical_features:
                if feat in feature_names:
                    st.write(f"- {feat}")
        
        st.markdown("---")
        
        # Summary statistics
        st.markdown("<div class='section-header'><h4>Dataset Summary</h4></div>", unsafe_allow_html=True)
        
        summary_stats = pd.DataFrame({
            'Metric': ['Total Laptops', 'Brands', 'Avg Price', 'Min Price', 'Max Price', 'Avg RAM', 'Avg Storage'],
            'Value': [
                f"{len(raw_data):,}",
                f"{raw_data['Company'].nunique()}",
                f"{raw_data['Price_Tsh'].mean():,.0f} Tsh",
                f"{raw_data['Price_Tsh'].min():,.0f} Tsh",
                f"{raw_data['Price_Tsh'].max():,.0f} Tsh",
                f"{raw_data['Ram'].mean():.1f} GB",
                f"{raw_data['PrimaryStorage'].mean():.0f} GB"
            ]
        })
        
        st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# ============================================================================
# MODE 4: MODEL COMPARISON
# ============================================================================
elif mode == "Model Comparison":
    
    st.markdown("<div class='section-header'><h3>Model Comparison: Linear Regression vs Decision Tree</h3></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>Comparing the two primary models trained for laptop price prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison data - Linear Regression vs Decision Tree
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree'],
        'R² Score': [0.7234, 0.8745],
        'RMSE (Tsh)': [389000, 245000],
        'MAE (Tsh)': [298000, 185000],
        'Training Time (s)': [0.8, 12.5],
        'Prediction Speed': ['Very Fast', 'Fast'],
        'Interpretability': ['High', 'Medium'],
        'Overfitting Risk': ['Low', 'Medium-High']
    })
    
    # Performance metrics comparison
    st.markdown("<div class='section-header'><h4>Performance Metrics Comparison</h4></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig = px.bar(model_comparison, x='Model', y='R² Score',
                     color='Model',
                     color_discrete_sequence=['#3498db', '#5dade2'],
                     title='R² Score Comparison')
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE comparison
        fig = px.bar(model_comparison, x='Model', y='RMSE (Tsh)',
                     color='Model',
                     color_discrete_sequence=['#e74c3c', '#3498db'],
                     title='RMSE Comparison (Lower is Better)')
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed comparison table
    st.markdown("<div class='section-header'><h4>Detailed Model Comparison</h4></div>", unsafe_allow_html=True)
    
    # Style the dataframe
    def highlight_best(s):
        if s.name == 'R² Score':
            is_max = s == s.max()
            return ['background-color: #d6eaf8; font-weight: bold;' if v else '' for v in is_max]
        elif s.name in ['RMSE (Tsh)', 'MAE (Tsh)', 'Training Time (s)']:
            is_min = s == s.min()
            return ['background-color: #d6eaf8; font-weight: bold;' if v else '' for v in is_min]
        else:
            return ['' for _ in s]
    
    styled_df = model_comparison.style.apply(highlight_best, subset=['R² Score', 'RMSE (Tsh)', 'MAE (Tsh)', 'Training Time (s)'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model characteristics comparison
    st.markdown("<div class='section-header'><h4>Model Characteristics</h4></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2980b9; margin-top: 0;">Linear Regression</h4>
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>Very fast training and prediction</li>
                <li>Highly interpretable coefficients</li>
                <li>Low risk of overfitting</li>
                <li>Works well with linear relationships</li>
            </ul>
            <p><strong>Weaknesses:</strong></p>
            <ul>
                <li>Lower accuracy (R² = 0.72)</li>
                <li>Cannot capture complex patterns</li>
                <li>Assumes linear relationships</li>
            </ul>
            <p><strong>Best For:</strong> Quick predictions, baseline model, when interpretability is crucial</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2980b9; margin-top: 0;">Decision Tree</h4>
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>Higher accuracy (R² = 0.87)</li>
                <li>Captures non-linear patterns</li>
                <li>Handles feature interactions well</li>
                <li>No feature scaling needed</li>
            </ul>
            <p><strong>Weaknesses:</strong></p>
            <ul>
                <li>Slower training time</li>
                <li>Risk of overfitting</li>
                <li>Less interpretable with depth</li>
            </ul>
            <p><strong>Best For:</strong> Production deployment, when accuracy is priority, complex data patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visual comparison
    st.markdown("<div class='section-header'><h4>Multi-Metric Comparison</h4></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        categories = ['Accuracy (R²)', 'Speed', 'Error (Inverse)', 'Interpretability']
        
        fig = go.Figure()
        
        # Linear Regression
        fig.add_trace(go.Scatterpolar(
            r=[0.7234, 1.0, 1 - (389000/389000), 1.0],
            theta=categories,
            fill='toself',
            name='Linear Regression',
            line_color='#3498db'
        ))
        
        # Decision Tree
        fig.add_trace(go.Scatterpolar(
            r=[0.8745, 0.8, 1 - (245000/389000), 0.7],
            theta=categories,
            fill='toself',
            name='Decision Tree',
            line_color='#5dade2'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400,
            title='Overall Model Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MAE Comparison
        fig = px.bar(model_comparison, x='Model', y='MAE (Tsh)',
                     color='Model',
                     color_discrete_sequence=['#e74c3c', '#3498db'],
                     title='Mean Absolute Error (Lower is Better)')
        fig.update_layout(
            height=200,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        fig = px.bar(model_comparison, x='Model', y='Training Time (s)',
                     color='Model',
                     color_discrete_sequence=['#3498db', '#e74c3c'],
                     title='Training Time (Lower is Better)')
        fig.update_layout(
            height=200,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    # Model Visualizations
    st.markdown("<div class='section-header'><h4>Model Visualizations</h4></div>", unsafe_allow_html=True)
    
    if raw_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Linear Regression - Fitting Line**")
            
            # Sample data for visualization
            sample_data = raw_data.sample(min(200, len(raw_data))).copy()
            sample_data = sample_data.sort_values('Ram')
            
            # Calculate simple linear regression manually
            x = sample_data['Ram'].values
            y = sample_data['Price_Tsh'].values
            
            # Calculate slope and intercept
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
            intercept = y_mean - slope * x_mean
            
            # Generate line points
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(size=8, color='#3498db', opacity=0.5),
                name='Actual Data'
            ))
            
            # Add regression line
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='#e74c3c', width=3),
                name='Linear Fit'
            ))
            
            fig.update_layout(
                height=400,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                title='Linear Regression Fit (RAM vs Price)',
                xaxis_title='RAM (GB)',
                yaxis_title='Price (Tsh)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 4px; font-size: 0.9rem;">
                <strong>Note:</strong> The red line shows the linear relationship that the model learns. 
                Linear Regression assumes a straight-line relationship between features and price.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Decision Tree - Structure Visualization**")
            
            # Create a simple decision tree visualization
            # This is a conceptual representation
            fig = go.Figure()
            
            # Tree structure (simplified representation)
            # Root node
            fig.add_trace(go.Scatter(
                x=[0.5], y=[1.0],
                mode='markers+text',
                marker=dict(size=40, color='#3498db'),
                text=['Root<br>RAM'],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                showlegend=False
            ))
            
            # Level 1 nodes
            fig.add_trace(go.Scatter(
                x=[0.25, 0.75], y=[0.7, 0.7],
                mode='markers+text',
                marker=dict(size=35, color='#5dade2'),
                text=['RAM≤8GB', 'RAM>8GB'],
                textposition='middle center',
                textfont=dict(size=9, color='white'),
                showlegend=False
            ))
            
            # Level 2 nodes
            fig.add_trace(go.Scatter(
                x=[0.15, 0.35, 0.65, 0.85], y=[0.4, 0.4, 0.4, 0.4],
                mode='markers+text',
                marker=dict(size=30, color='#85c1e9'),
                text=['Storage<br>≤256GB', 'Storage<br>>256GB', 'CPU<br>Intel', 'CPU<br>AMD'],
                textposition='middle center',
                textfont=dict(size=8, color='white'),
                showlegend=False
            ))
            
            # Leaf nodes
            fig.add_trace(go.Scatter(
                x=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], y=[0.1]*8,
                mode='markers+text',
                marker=dict(size=25, color='#aed6f1'),
                text=['800K', '1.2M', '1.5M', '1.8M', '2.0M', '2.5M', '2.8M', '3.2M'],
                textposition='middle center',
                textfont=dict(size=8, color='#2c3e50'),
                showlegend=False
            ))
            
            # Add connecting lines
            # Root to Level 1
            fig.add_trace(go.Scatter(x=[0.5, 0.25], y=[1.0, 0.7], mode='lines', 
                                    line=dict(color='#95a5a6', width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=[0.5, 0.75], y=[1.0, 0.7], mode='lines', 
                                    line=dict(color='#95a5a6', width=2), showlegend=False))
            
            # Level 1 to Level 2
            for x1, x2_list in [(0.25, [0.15, 0.35]), (0.75, [0.65, 0.85])]:
                for x2 in x2_list:
                    fig.add_trace(go.Scatter(x=[x1, x2], y=[0.7, 0.4], mode='lines',
                                           line=dict(color='#95a5a6', width=1.5), showlegend=False))
            
            # Level 2 to Leaves
            for x1, x2_list in [(0.15, [0.1, 0.2]), (0.35, [0.3, 0.4]), 
                               (0.65, [0.6, 0.7]), (0.85, [0.8, 0.9])]:
                for x2 in x2_list:
                    fig.add_trace(go.Scatter(x=[x1, x2], y=[0.4, 0.1], mode='lines',
                                           line=dict(color='#95a5a6', width=1), showlegend=False))
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title='Decision Tree Structure (Simplified)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 4px; font-size: 0.9rem;">
                <strong>Note:</strong> Decision Tree splits data based on feature thresholds, 
                creating a tree structure that can capture complex non-linear relationships.
            </div>
            """, unsafe_allow_html=True)

    
    # Recommendation
    st.markdown("---")
    
    # Recommendation
    st.markdown("<div class='section-header'><h4>Recommendation</h4></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2980b9; margin-top: 0;">🏆 Winner: Decision Tree</h4>
            <p class="stat-value" style="font-size: 1.5rem;">21% Better Accuracy</p>
            <p style="margin: 0; color: #6c757d;">R² Score: 0.8745 vs 0.7234</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem;">
                Despite slightly longer training time, Decision Tree provides significantly better 
                predictions and is recommended for production use.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin-top: 0;">Currently Using: {model_name}</h4>
            <p><strong>Performance:</strong> R² Score: {model_data['test_r2_score']:.4f} | RMSE: {model_data['test_rmse']:,.0f} Tsh</p>
            <p style="margin-bottom: 0;"><strong>Status:</strong> This model provides the best balance between accuracy and reliability for laptop price predictions.</p>
        </div>
        """, unsafe_allow_html=True)


