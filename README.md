# Laptop Price Prediction - AI Application

## 📋 Project Overview

This project implements a machine learning model to predict laptop prices based on various specifications. It includes a comprehensive Jupyter notebook for model development and a deployed Streamlit web application for real-time predictions.

## 🎯 Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Model Training**: Implementation of Linear Regression and Decision Tree models
- **Model Evaluation**: Detailed performance comparison using multiple metrics
- **Interactive Web App**: User-friendly interface for price predictions
- **Visualizations**: Multiple plots for data analysis and model performance
- **Budget analysis**: Suggest Laptops based on budget

## 📊 Model Performance

- **Best Model**: Decision Tree Regressor
- **R² Score**: 0.7718
- **RMSE**: 959,110.84 Tsh
- **Training Data**: 1,020 samples
- **Test Data**: 255 samples

## 📁 Project Structure

```
laptop-price-prediction/
│
├── ml_project.ipynb          # Jupyter notebook with complete ML workflow
├── app.py                     # Streamlit application
├── model.pkl                  # Trained model (pickle file)
├── train_model.py            # Script to train the model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── laptop_prices.csv         # Dataset (place your data here)
```

## 🚀 Deployment Instructions

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Create a GitHub Repository**
   ```bash
   git init
   git add app.py model.pkl requirements.txt
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/laptop-price-predictor.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Access Your App**
   - Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

### Option 2: Deploy using Flask

1. **Create Flask App** (alternative to Streamlit)
   ```python
   # See flask_app.py for implementation
   ```

2. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access Locally**
   - Open browser at: `http://localhost:8501`

## 💻 Using the Jupyter Notebook

1. **Install Jupyter**
   ```bash
   pip install jupyter notebook
   ```

2. **Run the Notebook**
   ```bash
   jupyter notebook ml_project.ipynb
   ```

3. **Execute All Cells**
   - Run all cells to see the complete ML workflow
   - Visualizations will be generated
   - Model will be saved as `model.pkl`

## 🔧 Model Training

To retrain the model with new data:

```bash
python train_model.py
```

This will:
- Load the dataset
- Preprocess the data
- Train both Linear Regression and Decision Tree models
- Evaluate and compare models
- Save the best model as `model.pkl`

## 📊 Features Used

The model uses the following 13 features:

1. **Company**: Laptop manufacturer (Apple, HP, Dell, etc.)
2. **TypeName**: Type of laptop (Ultrabook, Notebook, Gaming, etc.)
3. **Inches**: Screen size in inches
4. **Ram**: RAM capacity in GB
5. **OS**: Operating system
6. **Weight**: Laptop weight in kg
7. **Touchscreen**: Whether it has touchscreen (Yes/No)
8. **IPSpanel**: IPS display panel (Yes/No)
9. **RetinaDisplay**: Retina display (Yes/No)
10. **CPU_company**: CPU manufacturer (Intel, AMD)
11. **CPU_freq**: CPU frequency in GHz
12. **PrimaryStorage**: Storage capacity in GB
13. **GPU_company**: GPU manufacturer (Intel, AMD, Nvidia)

## 📈 Model Comparison

| Model | Train R² | Test R² | Train RMSE | Test RMSE |
|-------|----------|---------|------------|-----------|
| Linear Regression | 0.7156 | 0.6944 | 1,067,938 | 1,109,884 |
| Decision Tree | 0.9483 | 0.7718 | 456,107 | 959,111 |

**Winner**: Decision Tree (higher R² score on test set)

## 🎨 Application Features

### Web Interface
- Clean, modern UI with responsive design
- Real-time price predictions
- Detailed prediction breakdown
- Model performance metrics
- Interactive input fields

### Input Options
- Dropdown menus for categorical features
- Sliders for numerical features
- Radio buttons for binary options
- Smart defaults based on common configurations

## 📝 Example Usage

### Using the Web App

1. Select laptop specifications:
   - Company: Dell
   - Type: Ultrabook
   - Screen: 15.6 inches
   - RAM: 16 GB

2. Configure hardware:
   - CPU: Intel @ 2.5 GHz
   - GPU: Nvidia
   - Storage: 512 GB

3. Click "Predict Price"

4. View predicted price and details

### Using Python

```python
import pickle
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Prepare input
input_data = pd.DataFrame({
    'Company': [5],  # Encoded value
    'TypeName': [4],
    'Inches': [15.6],
    'Ram': [16],
    # ... other features
})

# Scale and predict
scaled_input = model_data['scaler'].transform(input_data)
prediction = model_data['model'].predict(scaled_input)
print(f"Predicted Price: {prediction[0]:,.0f} Tsh")
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model file not found**
   - Ensure `model.pkl` is in the same directory as `app.py`
   - Run `train_model.py` to generate the model

3. **Streamlit errors**
   ```bash
   streamlit cache clear
   streamlit run app.py
   ```

## 📚 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- matplotlib
- seaborn

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Edwin Laswai 

## 🙏 Acknowledgments

- Dataset source: Laptop prices dataset
- Built with Streamlit, scikit-learn, and pandas
- Inspired by real-world ML deployment challenges

---

**Note**: For production deployment, ensure you have the necessary data privacy and security measures in place.
