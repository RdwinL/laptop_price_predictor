import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = 0
if 'prediction_inputs' not in st.session_state:
    st.session_state.prediction_inputs = {}

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'model.pkl' is in the same directory.")
        st.stop()

# Load the raw dataset for dynamic filtering
@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv('laptop_prices.csv')
        return df
    except FileNotFoundError:
        # Fallback to default options if CSV not found
        return None

# Load model and data
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
model_name = model_data['model_name']
raw_data = load_raw_data()

# Create brand-specific mappings from raw data
@st.cache_data
def get_brand_mappings():
    if raw_data is not None:
        # Brand to OS mapping
        brand_os = raw_data.groupby('Company')['OS'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        
        # Brand to TypeName mapping
        brand_type = raw_data.groupby('Company')['TypeName'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        
        # Brand to CPU company mapping
        brand_cpu = raw_data.groupby('Company')['CPU_company'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        
        # Brand to GPU company mapping
        brand_gpu = raw_data.groupby('Company')['GPU_company'].apply(lambda x: sorted(x.unique().tolist())).to_dict()
        
        return brand_os, brand_type, brand_cpu, brand_gpu
    else:
        # Fallback defaults
        companies = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Toshiba', 'Samsung']
        brand_os = {c: ['Windows 10', 'macOS', 'No OS', 'Linux'] for c in companies}
        brand_os['Apple'] = ['macOS', 'Mac OS X']
        brand_type = {c: ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'] for c in companies}
        brand_cpu = {c: ['Intel', 'AMD'] for c in companies}
        brand_gpu = {c: ['Intel', 'AMD', 'Nvidia'] for c in companies}
        return brand_os, brand_type, brand_cpu, brand_gpu

brand_os_map, brand_type_map, brand_cpu_map, brand_gpu_map = get_brand_mappings()

# Get all unique companies
all_companies = sorted(brand_os_map.keys())

# Title and description
st.title("💻 Laptop Price Prediction System")
st.markdown(f"""
    <div class="info-box">
        <h4>Welcome to the AI-Powered Laptop Price Predictor!</h4>
        <p>This application uses a <b>{model_name}</b> model trained on 13 features 
        to predict laptop prices in Tanzanian Shillings (Tsh).</p>
        <p><b>Model Performance:</b> R² Score = {model_data['test_r2_score']:.4f} | 
        RMSE = {model_data['test_rmse']:,.0f} Tsh</p>
    </div>
""", unsafe_allow_html=True)


# Create expandable sections for better organization
with st.expander("📋 **STEP 1: Select Laptop Brand**", expanded=True):
    company = st.selectbox(
        "Choose the laptop manufacturer",
        all_companies,
        key='company_select',
        help="Select the brand of laptop you want to price"
    )

# Get brand-specific options
available_os = brand_os_map.get(company, ['Windows 10'])
available_types = brand_type_map.get(company, ['Notebook'])
available_cpu = brand_cpu_map.get(company, ['Intel'])
available_gpu = brand_gpu_map.get(company, ['Intel'])

with st.expander("💻 **STEP 2: Configure Laptop Specifications**", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        type_name = st.selectbox(
            "Laptop Type",
            available_types,
            key='type_select',
            help=f"Available types for {company}"
        )
        
        os = st.selectbox(
            "Operating System",
            available_os,
            key='os_select',
            help=f"OS options available for {company}"
        )
        
        ram = st.selectbox(
            "RAM (GB)",
            [2, 4, 6, 8, 12, 16, 24, 32, 64],
            index=3,  # Default to 8GB
            key='ram_select'
        )
        
        primary_storage = st.selectbox(
            "Storage (GB)",
            [32, 64, 128, 256, 512, 1024, 2048],
            index=3,  # Default to 256GB
            key='storage_select'
        )
    
    with col2:
        cpu_company = st.selectbox(
            "CPU Brand",
            available_cpu,
            key='cpu_select',
            help=f"CPU options for {company}"
        )
        
        gpu_company = st.selectbox(
            "GPU Brand",
            available_gpu,
            key='gpu_select',
            help=f"GPU options for {company}"
        )
        
        cpu_freq = st.number_input(
            "CPU Frequency (GHz)",
            min_value=1.0,
            max_value=4.0,
            value=2.5,
            step=0.1,
            key='cpu_freq_input'
        )
        
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key='weight_input'
        )

with st.expander("🖥️ **STEP 3: Display & Physical Features**", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        inches = st.number_input(
            "Screen Size (inches)",
            min_value=10.0,
            max_value=18.0,
            value=15.6,
            step=0.1,
            key='inches_input'
        )
        
        touchscreen = st.radio(
            "Touchscreen",
            ["No", "Yes"],
            horizontal=True,
            key='touch_select'
        )
        
    with col2:
        ips_panel = st.radio(
            "IPS Panel",
            ["No", "Yes"],
            horizontal=True,
            key='ips_select'
        )
        
        retina_display = st.radio(
            "Retina Display",
            ["No", "Yes"],
            horizontal=True,
            key='retina_select'
        )


# Prediction button with session state management
st.markdown("---")
col_predict, col_reset = st.columns([3, 1])

with col_predict:
    predict_button = st.button("🔮 Predict Price", use_container_width=True)

with col_reset:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.prediction_made = False
        st.session_state.predicted_price = 0
        st.session_state.prediction_inputs = {}
        st.rerun()

if predict_button:
    try:
        # Store current inputs in session state
        current_inputs = {
            'company': company,
            'type_name': type_name,
            'inches': inches,
            'ram': ram,
            'os': os,
            'weight': weight,
            'touchscreen': touchscreen,
            'ips_panel': ips_panel,
            'retina_display': retina_display,
            'cpu_company': cpu_company,
            'cpu_freq': cpu_freq,
            'primary_storage': primary_storage,
            'gpu_company': gpu_company
        }
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Inches': [inches],
            'Ram': [ram],
            'OS': [os],
            'Weight': [weight],
            'Touchscreen': [1 if touchscreen == "Yes" else 0],
            'IPSpanel': [1 if ips_panel == "Yes" else 0],
            'RetinaDisplay': [1 if retina_display == "Yes" else 0],
            'CPU_company': [cpu_company],
            'CPU_freq': [cpu_freq],
            'PrimaryStorage': [primary_storage],
            'GPU_company': [gpu_company]
        })
        
        # Encode categorical variables
        for col in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']:
            le = label_encoders[col]
            try:
                input_data[col] = le.transform(input_data[col])
            except ValueError:
                # If value not in training data, use the most common class
                input_data[col] = 0
        
        # Ensure correct column order
        input_data = input_data[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Store prediction in session state
        st.session_state.prediction_made = True
        st.session_state.predicted_price = prediction
        st.session_state.prediction_inputs = current_inputs
        
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.error("Please check your inputs and try again.")


# Display prediction if available
if st.session_state.prediction_made:
    prediction = st.session_state.predicted_price
    saved_inputs = st.session_state.prediction_inputs
    
    # Get similar products from the dataset
    def get_product_recommendations(brand, predicted_price, top_n=3):
        if raw_data is not None:
            # Filter by brand
            brand_laptops = raw_data[raw_data['Company'] == brand].copy()
            
            if len(brand_laptops) > 0:
                # Calculate price difference
                brand_laptops['price_diff'] = abs(brand_laptops['Price_Tsh'] - predicted_price)
                
                # Sort by price similarity
                similar_products = brand_laptops.nsmallest(top_n, 'price_diff')
                
                recommendations = []
                for _, row in similar_products.iterrows():
                    recommendations.append({
                        'name': f"{row['Company']} {row['Product']}",
                        'type': row['TypeName'],
                        'price': row['Price_Tsh'],
                        'ram': row['Ram'],
                        'storage': row['PrimaryStorage'],
                        'screen': row['Inches'],
                        'cpu': f"{row['CPU_company']} {row['CPU_model']}",
                        'os': row['OS']
                    })
                
                return recommendations
        return []
    
    # Display prediction
    st.markdown(f"""
        <div class="prediction-box">
            <h2>💰 Predicted Laptop Price</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:,.0f} Tsh</h1>
            <p style="font-size: 1.2rem;">≈ ${prediction/2500:.2f} USD (approximate)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get product recommendations
    recommendations = get_product_recommendations(saved_inputs['company'], prediction, top_n=3)
    
    # Display recommended products
    if recommendations:
        st.markdown("### 🎯 Suggested Products from " + saved_inputs['company'])
        st.markdown("*Based on your predicted price, here are similar laptops from our dataset:*")
        
        for i, product in enumerate(recommendations, 1):
            with st.container():
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        padding: 1.5rem;
                        border-radius: 10px;
                        margin: 1rem 0;
                        border-left: 5px solid #667eea;
                    ">
                        <h4 style="margin: 0 0 1rem 0; color: #333;">
                            {i}. {product['name']}
                        </h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                            <div>
                                <p style="margin: 0.3rem 0;"><b>💰 Price:</b> {product['price']:,.0f} Tsh (≈ ${product['price']/2500:.0f})</p>
                                <p style="margin: 0.3rem 0;"><b>💻 Type:</b> {product['type']}</p>
                                <p style="margin: 0.3rem 0;"><b>🖥️ OS:</b> {product['os']}</p>
                            </div>
                            <div>
                                <p style="margin: 0.3rem 0;"><b>🧠 RAM:</b> {product['ram']} GB</p>
                                <p style="margin: 0.3rem 0;"><b>💾 Storage:</b> {product['storage']} GB</p>
                                <p style="margin: 0.3rem 0;"><b>📺 Screen:</b> {product['screen']}" | <b>⚙️ CPU:</b> {product['cpu']}</p>
                            </div>
                        </div>
                        <p style="margin: 1rem 0 0 0; color: #666; font-size: 0.9rem;">
                            <b>Price Difference:</b> {abs(product['price'] - prediction):,.0f} Tsh from your predicted price
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.info("💡 **Note:** These are actual laptops from our dataset with prices closest to your prediction.")
    
    # Additional information
    st.success("✅ Prediction completed successfully!")
    
    # Show prediction details in expandable section
    with st.expander("📊 View Your Input Specifications"):
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.write("**Input Features:**")
            st.write(f"- Brand: {saved_inputs['company']}")
            st.write(f"- Type: {saved_inputs['type_name']}")
            st.write(f"- Screen: {saved_inputs['inches']}\"")
            st.write(f"- RAM: {saved_inputs['ram']} GB")
            st.write(f"- Storage: {saved_inputs['primary_storage']} GB")
            st.write(f"- Weight: {saved_inputs['weight']} kg")
        
        with col_detail2:
            st.write("**Configuration:**")
            st.write(f"- OS: {saved_inputs['os']}")
            st.write(f"- CPU: {saved_inputs['cpu_company']} @ {saved_inputs['cpu_freq']} GHz")
            st.write(f"- GPU: {saved_inputs['gpu_company']}")
            st.write(f"- Touchscreen: {saved_inputs['touchscreen']}")
            st.write(f"- IPS Panel: {saved_inputs['ips_panel']}")
            st.write(f"- Retina Display: {saved_inputs['retina_display']}")
    
    # Price range information
    st.info(f"""
        ℹ️ **Model Info:** This prediction is based on {model_name} model with 
        R² score of {model_data['test_r2_score']:.4f}. Actual prices may vary 
        based on market conditions, promotions, and availability.
    """)

# Sidebar with additional information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/610/610413.png", width=100)
    st.title("About")
    st.markdown("""
        ### 📌 Model Information
        
        **Model Type:** {0}
        
        **Performance Metrics:**
        - R² Score: {1:.4f}
        - RMSE: {2:,.0f} Tsh
        
        **Features Used:** {3}
        
        ---
        
        ### 💡 How to Use
        1. Select laptop specifications
        2. Configure hardware options
        3. Click "Predict Price"
        4. View the estimated price
        
        ---
        
        ### 📊 Dataset Info
        - Total Laptops: 1,275
        - Features: 13
        - Price Range: 100K - 10M Tsh
        
        ---
        
        ### 🎯 Accuracy
        The model achieves **{1:.1f}%** accuracy 
        in predicting laptop prices based on 
        the R² score metric.
        
        ---
        
        Made with ❤️ using Streamlit
    """.format(
        model_name,
        model_data['test_r2_score'],
        model_data['test_rmse'],
        len(feature_names)
    ))
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("For questions or feedback, please contact the development team.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Laptop Price Prediction System v1.0 | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
