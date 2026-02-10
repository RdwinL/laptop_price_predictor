from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']

@app.route('/')
def home():
    return render_template('index.html', 
                         model_name=model_data['model_name'],
                         r2_score=model_data['test_r2_score'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Company': [data['company']],
            'TypeName': [data['type_name']],
            'Inches': [float(data['inches'])],
            'Ram': [int(data['ram'])],
            'OS': [data['os']],
            'Weight': [float(data['weight'])],
            'Touchscreen': [1 if data['touchscreen'] == 'Yes' else 0],
            'IPSpanel': [1 if data['ips_panel'] == 'Yes' else 0],
            'RetinaDisplay': [1 if data['retina_display'] == 'Yes' else 0],
            'CPU_company': [data['cpu_company']],
            'CPU_freq': [float(data['cpu_freq'])],
            'PrimaryStorage': [int(data['primary_storage'])],
            'GPU_company': [data['gpu_company']]
        })
        
        # Encode categorical variables
        for col in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']:
            le = label_encoders[col]
            try:
                input_data[col] = le.transform(input_data[col])
            except ValueError:
                input_data[col] = 0
        
        # Ensure correct column order
        input_data = input_data[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'prediction_tsh': f"{prediction:,.0f} Tsh",
            'prediction_usd': f"${prediction/2500:.2f} USD"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        
        # Similar processing as above
        input_data = pd.DataFrame({
            'Company': [data['company']],
            'TypeName': [data['type_name']],
            'Inches': [float(data['inches'])],
            'Ram': [int(data['ram'])],
            'OS': [data['os']],
            'Weight': [float(data['weight'])],
            'Touchscreen': [1 if data.get('touchscreen', 'No') == 'Yes' else 0],
            'IPSpanel': [1 if data.get('ips_panel', 'No') == 'Yes' else 0],
            'RetinaDisplay': [1 if data.get('retina_display', 'No') == 'Yes' else 0],
            'CPU_company': [data['cpu_company']],
            'CPU_freq': [float(data['cpu_freq'])],
            'PrimaryStorage': [int(data['primary_storage'])],
            'GPU_company': [data['gpu_company']]
        })
        
        # Encode categorical variables
        for col in ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']:
            le = label_encoders[col]
            try:
                input_data[col] = le.transform(input_data[col])
            except ValueError:
                input_data[col] = 0
        
        # Ensure correct column order
        input_data = input_data[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction_tsh': float(prediction),
            'prediction_usd': float(prediction / 2500),
            'model': model_data['model_name'],
            'confidence': model_data['test_r2_score']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
