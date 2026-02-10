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

# Load model
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
model_name = model_data['model_name']

# Title and description
st.title("💻 Laptop Price Prediction System")
st.markdown(f"""
    <div class="info-box">
        <h4>Welcome to the AI-Powered Laptop Price Predictor!</h4>
        <p>This application uses a <b>{model_name}</b> model trained on {len(feature_names)} features 
        to predict laptop prices in Tanzanian Shillings (Tsh).</p>
        <p><b>Model Performance:</b> R² Score = {model_data['test_r2_score']:.4f} | 
        RMSE = {model_data['test_rmse']:,.0f} Tsh</p>
    </div>
""", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

# Get unique values for categorical features
companies = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Toshiba', 'Samsung', 
             'Razer', 'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 
             'LG', 'Huawei', 'Other']

type_names = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook']

os_options = ['Windows 10', 'macOS', 'No OS', 'Windows 7', 'Mac OS X', 'Linux', 'Chrome OS', 'Windows 10 S']

cpu_companies = ['Intel', 'AMD', 'Samsung']

gpu_companies = ['Intel', 'AMD', 'Nvidia']

# Input fields in left column
with col1:
    st.subheader("📋 Laptop Specifications")
    
    company = st.selectbox("Company/Brand", companies)
    
    type_name = st.selectbox("Laptop Type", type_names)
    
    inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)
    
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
    
    os = st.selectbox("Operating System", os_options)
    
    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

# Input fields in right column
with col2:
    st.subheader("⚙️ Hardware Configuration")
    
    touchscreen = st.radio("Touchscreen", ["No", "Yes"])
    
    ips_panel = st.radio("IPS Panel", ["No", "Yes"])
    
    retina_display = st.radio("Retina Display", ["No", "Yes"])
    
    cpu_company = st.selectbox("CPU Brand", cpu_companies)
    
    cpu_freq = st.slider("CPU Frequency (GHz)", 1.0, 4.0, 2.5, 0.1)
    
    primary_storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048])
    
    gpu_company = st.selectbox("GPU Brand", gpu_companies)

# Prediction button
if st.button("🔮 Predict Price"):
    try:
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
        
        # Display prediction
        st.markdown(f"""
            <div class="prediction-box">
                <h2>💰 Predicted Laptop Price</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:,.0f} Tsh</h1>
                <p style="font-size: 1.2rem;">≈ ${prediction/2500:.2f} USD (approximate)</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional information
        st.success("✅ Prediction completed successfully!")
        
        # Show prediction details in expandable section
        with st.expander("📊 View Prediction Details"):
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.write("**Input Features:**")
                st.write(f"- Brand: {company}")
                st.write(f"- Type: {type_name}")
                st.write(f"- Screen: {inches}\"")
                st.write(f"- RAM: {ram} GB")
                st.write(f"- Storage: {primary_storage} GB")
                st.write(f"- Weight: {weight} kg")
            
            with col_detail2:
                st.write("**Configuration:**")
                st.write(f"- OS: {os}")
                st.write(f"- CPU: {cpu_company} @ {cpu_freq} GHz")
                st.write(f"- GPU: {gpu_company}")
                st.write(f"- Touchscreen: {touchscreen}")
                st.write(f"- IPS Panel: {ips_panel}")
                st.write(f"- Retina Display: {retina_display}")
        
        # Price range information
        st.info(f"""
            ℹ️ **Note:** This prediction is based on {model_name} model with 
            R² score of {model_data['test_r2_score']:.4f}. Actual prices may vary 
            based on market conditions, promotions, and availability.
        """)
        
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.error("Please check your inputs and try again.")

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
