from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import json

app = Flask(__name__)

# Load model and data
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

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

@app.route('/')
def index():
    return render_template('index.html', 
                         companies=all_companies,
                         model_name=model_name,
                         r2_score=model_data['test_r2_score'],
                         rmse=model_data['test_rmse'],
                         features=len(feature_names))

@app.route('/get_brand_options', methods=['POST'])
def get_brand_options():
    company = request.json.get('company')
    return jsonify({
        'os': brand_os_map.get(company, ['Windows 10']),
        'types': brand_type_map.get(company, ['Notebook']),
        'cpu': brand_cpu_map.get(company, ['Intel']),
        'gpu': brand_gpu_map.get(company, ['Intel'])
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_data = pd.DataFrame({
            'Company': [data['company']],
            'TypeName': [data['type']],
            'Inches': [float(data['inches'])],
            'Ram': [int(data['ram'])],
            'OS': [data['os']],
            'Weight': [float(data['weight'])],
            'Touchscreen': [1 if data['touchscreen'] == 'Yes' else 0],
            'IPSpanel': [1 if data['ips_panel'] == 'Yes' else 0],
            'RetinaDisplay': [1 if data['retina_display'] == 'Yes' else 0],
            'CPU_company': [data['cpu_company']],
            'CPU_freq': [float(data['cpu_freq'])],
            'PrimaryStorage': [int(data['storage'])],
            'GPU_company': [data['gpu_company']]
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
        
        # Get recommendations
        recommendations = []
        if raw_data is not None:
            brand_laptops = raw_data[raw_data['Company'] == data['company']].copy()
            if len(brand_laptops) > 0:
                brand_laptops['price_diff'] = abs(brand_laptops['Price_Tsh'] - prediction)
                top_3 = brand_laptops.nsmallest(3, 'price_diff')
                
                for _, row in top_3.iterrows():
                    recommendations.append({
                        'name': f"{row['Company']} {row['Product']}",
                        'price': int(row['Price_Tsh']),
                        'type': row['TypeName'],
                        'ram': int(row['Ram']),
                        'storage': int(row['PrimaryStorage']),
                        'screen': float(row['Inches']),
                        'difference': int(abs(row['Price_Tsh'] - prediction))
                    })
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_usd': round(prediction / 2500, 2),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/budget_search', methods=['POST'])
def budget_search():
    try:
        data = request.json
        min_price = int(data['min_price'])
        max_price = int(data['max_price'])
        brand_filter = data.get('brands', [])
        type_filter = data.get('types', [])
        min_ram = int(data.get('min_ram', 0))
        
        results = raw_data[(raw_data['Price_Tsh'] >= min_price) & (raw_data['Price_Tsh'] <= max_price)]
        
        if brand_filter:
            results = results[results['Company'].isin(brand_filter)]
        if type_filter:
            results = results[results['TypeName'].isin(type_filter)]
        if min_ram > 0:
            results = results[results['Ram'] >= min_ram]
        
        results = results.sort_values('Price_Tsh')
        
        laptops = []
        for _, laptop in results.head(15).iterrows():
            laptops.append({
                'name': f"{laptop['Company']} {laptop['Product']}",
                'price': int(laptop['Price_Tsh']),
                'price_usd': int(laptop['Price_Tsh'] / 2500),
                'type': laptop['TypeName'],
                'os': laptop['OS'],
                'ram': int(laptop['Ram']),
                'storage': int(laptop['PrimaryStorage']),
                'screen': float(laptop['Inches']),
                'cpu': laptop['CPU_company']
            })
        
        stats = {
            'count': len(results),
            'avg_price': int(results['Price_Tsh'].mean()) if len(results) > 0 else 0,
            'avg_ram': int(results['Ram'].mean()) if len(results) > 0 else 0,
            'avg_storage': int(results['PrimaryStorage'].mean()) if len(results) > 0 else 0,
            'avg_screen': round(results['Inches'].mean(), 1) if len(results) > 0 else 0
        }
        
        return jsonify({
            'success': True,
            'laptops': laptops,
            'stats': stats,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analytics')
def analytics():
    if raw_data is None:
        return jsonify({'error': 'Data not available'})
    
    # Price distribution
    price_hist = px.histogram(raw_data, x='Price_Tsh', nbins=40)
    price_hist_json = price_hist.to_json()
    
    # RAM distribution
    ram_counts = raw_data['Ram'].value_counts().sort_index()
    ram_chart = px.bar(x=ram_counts.index, y=ram_counts.values)
    ram_chart_json = ram_chart.to_json()
    
    # Price by brand
    brand_prices = raw_data.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=True).tail(10)
    brand_chart = px.bar(x=brand_prices.values, y=brand_prices.index, orientation='h')
    brand_chart_json = brand_chart.to_json()
    
    return render_template('analytics.html',
                         price_chart=price_hist_json,
                         ram_chart=ram_chart_json,
                         brand_chart=brand_chart_json,
                         model_name=model_name,
                         r2_score=model_data['test_r2_score'],
                         rmse=model_data['test_rmse'],
                         features=len(feature_names))

@app.route('/comparison')
def comparison():
    return render_template('comparison.html',
                         model_name=model_name,
                         r2_score=model_data['test_r2_score'],
                         rmse=model_data['test_rmse'])

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        data = request.json
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 15, 'LAPTOP PRICE PREDICTION REPORT', 0, 1, 'C')
        pdf.ln(5)
        
        # Prediction Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(52, 152, 219)
        pdf.cell(0, 10, 'Prediction Summary', 0, 1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(80, 8, 'Predicted Price (Tsh):', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"{data['prediction']:,} Tsh", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(80, 8, 'Predicted Price (USD):', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"${data['prediction_usd']} USD", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(80, 8, 'Prediction Date:', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
        pdf.ln(5)
        
        # Input Specifications
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(52, 152, 219)
        pdf.cell(0, 10, 'Input Specifications', 0, 1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        specs = data['specs']
        pdf.set_text_color(0, 0, 0)
        for key, value in specs.items():
            pdf.set_font('Arial', '', 10)
            pdf.cell(80, 7, key + ':', 0, 0)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 7, str(value), 0, 1)
        
        pdf.ln(5)
        
        # Model Information
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(52, 152, 219)
        pdf.cell(0, 10, 'Model Information', 0, 1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(80, 8, 'Model Type:', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, model_name, 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(80, 8, 'R² Score:', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"{model_data['test_r2_score']:.4f}", 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(80, 8, 'RMSE:', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"{model_data['test_rmse']:,.0f} Tsh", 0, 1)
        
        # Footer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, 'Generated by Laptop Price Predictor', 0, 1, 'C')
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        return send_file(
            BytesIO(pdf_output),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"laptop_price_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
