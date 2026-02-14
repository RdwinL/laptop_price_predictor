# Laptop Price Predictor - Flask App

AI/ML-powered laptop price prediction and analysis system built with Flask.

## Features

- 🤖 Machine Learning price prediction with 87% accuracy
- 💰 Budget-based laptop search and filtering
- 📊 Interactive analytics and visualizations
- 📥 PDF report generation
- 🔍 Model comparison (Linear Regression vs Decision Tree)
- 🐳 Docker support for easy deployment

## Tech Stack

- **Backend**: Flask, Python 3.11
- **ML**: scikit-learn, pandas, numpy
- **Visualization**: Plotly
- **Frontend**: Bootstrap 5, JavaScript
- **PDF Generation**: FPDF2
- **Deployment**: Docker, Gunicorn

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- `model.pkl` (trained ML model)
- `laptop_prices.csv` (dataset)

## Installation

### Option 1: Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd flask_app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add required files**
- Place `model.pkl` in the root directory
- Place `laptop_prices.csv` in the root directory

5. **Run the application**
```bash
python app.py
```

6. **Access the app**
Open browser and navigate to: `http://localhost:5000`

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **Access the app**
Open browser and navigate to: `http://localhost:5000`

3. **View logs**
```bash
docker-compose logs -f
```

4. **Stop the application**
```bash
docker-compose down
```

### Option 3: Docker (Manual)

1. **Build the image**
```bash
docker build -t laptop-price-predictor .
```

2. **Run the container**
```bash
docker run -d -p 5000:5000 \
  -v $(pwd)/model.pkl:/app/model.pkl \
  -v $(pwd)/laptop_prices.csv:/app/laptop_prices.csv \
  --name laptop-predictor \
  laptop-price-predictor
```

3. **Access the app**
Open browser and navigate to: `http://localhost:5000`

## Project Structure

```
flask_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Docker ignore file
├── templates/            # HTML templates
│   ├── index.html
│   ├── analytics.html
│   └── comparison.html
├── static/               # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── model.pkl            # ML model (required)
└── laptop_prices.csv    # Dataset (required)
```

## API Endpoints

### GET /
Main prediction interface

### POST /get_brand_options
Get available options for selected brand
```json
{
  "company": "Apple"
}
```

### POST /predict
Predict laptop price
```json
{
  "company": "Apple",
  "type": "Notebook",
  "inches": 15.6,
  "ram": 8,
  "storage": 256,
  "os": "macOS",
  "weight": 2.0,
  "touchscreen": "No",
  "ips_panel": "Yes",
  "retina_display": "Yes",
  "cpu_company": "Intel",
  "cpu_freq": 2.5,
  "gpu_company": "Intel"
}
```

### POST /budget_search
Search laptops by budget
```json
{
  "min_price": 1000000,
  "max_price": 3000000,
  "min_ram": 8
}
```

### POST /download_report
Download prediction report as PDF

### GET /analytics
View analytics dashboard

### GET /comparison
View model comparison

## Environment Variables

- `FLASK_APP`: Application entry point (default: app.py)
- `FLASK_ENV`: Environment mode (development/production)

## Production Deployment

For production deployment:

1. **Set environment to production**
```bash
export FLASK_ENV=production
```

2. **Use Gunicorn**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

3. **Deploy with Docker** (recommended)
```bash
docker-compose up -d
```

## Model Information

- **Algorithm**: Decision Tree Regressor
- **R² Score**: 0.8745
- **RMSE**: ~245,000 Tsh
- **Features**: 13 input features
- **Dataset**: 1,275 laptop samples

## Screenshots

(Add screenshots of your application here)

## License

MIT License

## Author

AI/ML Engineer | Data Scientist

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## Support

For support, email your-email@example.com or open an issue on GitHub.
