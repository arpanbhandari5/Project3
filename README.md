# Telecom Churn Prediction System

An end-to-end machine learning system for predicting telecom customer churn with ensemble models, SHAP explainability, and real-time drift monitoring.

## 🌟 Features

### 1. **End-to-End Classification Flow**
- **User Input → Preprocessing → Ensemble → Classification**
- Seamless data flow from raw input to final prediction
- Automated feature engineering and transformation

### 2. **Ensemble Models**
- **XGBoost**: Gradient boosting with 40% weight
- **LightGBM**: Fast gradient boosting with 35% weight
- **Neural Network**: Deep learning with 25% weight
- Weighted voting for robust predictions

### 3. **Parallel SHAP Explainability**
- Model-agnostic explanations using SHAP values
- Parallel computation for faster processing
- Feature importance and waterfall plots
- Per-prediction explanations

### 4. **PSI Monitoring with Drift Detection**
- Population Stability Index (PSI) calculation
- Background monitoring with configurable intervals
- Automatic drift detection with thresholds:
  - **Warning**: PSI > 0.1
  - **Critical**: PSI > 0.25
- Real-time alerts for data drift

### 5. **Admin Dashboard**
- Web-based monitoring interface
- Real-time system status
- Drift alerts visualization
- Prediction history tracking
- REST API endpoints

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
python main.py --demo
```

This will:
1. Generate sample telecom churn data
2. Train the ensemble models (XGBoost, LightGBM, Neural Network)
3. Initialize SHAP explainer
4. Set up PSI monitoring
5. Make sample predictions with explanations
6. Check for data drift

### Launch Dashboard

```bash
python main.py --dashboard
```

Access the dashboard at: `http://localhost:5000`

## 📊 Architecture

```
User Input
    ↓
[Preprocessing]
    ↓
[Ensemble Models]
    ├─ XGBoost (40%)
    ├─ LightGBM (35%)
    └─ Neural Network (25%)
    ↓
[Weighted Classification]
    ↓
Final Prediction

    ⇄ [SHAP Explainer] (Parallel)
    ⇄ [PSI Monitor] (Background)
    ⇄ [Admin Dashboard] (Alerts)
```

## 🔧 Usage Examples

### Training the Pipeline

```python
from main import ChurnPredictionPipeline
import pandas as pd

# Load your data
X_train = pd.DataFrame(...)  # Features
y_train = ...  # Target labels

# Initialize and train
pipeline = ChurnPredictionPipeline()
pipeline.train(X_train, y_train)

# Save models
pipeline.save()
```

### Making Predictions

```python
# Single prediction with explanation
user_input = {
    'tenure': 12,
    'monthly_charges': 75.5,
    'contract_type': 'Month-to-month',
    # ... other features
}

result = pipeline.predict_single(user_input, explain=True)
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Top Features: {result['top_features']}")
```

### Monitoring Data Drift

```python
# Check drift manually
drift_status = pipeline.monitor_drift(X_current)
print(f"Status: {drift_status['overall_status']}")

# Start background monitoring
def get_current_data():
    # Your data source
    return current_data

pipeline.start_background_monitoring(get_current_data, check_interval=3600)
```

### Launching Dashboard

```python
pipeline.launch_dashboard(host='0.0.0.0', port=5000)
```

## 📡 API Endpoints

### Dashboard API

- `GET /` - Dashboard home page
- `GET /api/alerts` - Get all alerts
- `GET /api/alerts/<severity>` - Get alerts by severity
- `GET /api/drift/status` - Current drift status
- `GET /api/drift/history` - Drift check history
- `GET /api/predictions/history` - Prediction history
- `POST /api/predict` - Make prediction
- `GET /api/system/info` - System information

### Example API Call

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [12, 75.5, 900, 0, 1, 0, 1, 0, 1, 0]}'
```

## 🧪 Testing

Run tests:

```bash
python -m pytest tests/
```

Or run specific test modules:

```bash
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_models.py
python -m pytest tests/test_monitoring.py
```

## 📁 Project Structure

```
Project3/
├── config.py                 # Configuration settings
├── main.py                   # Main orchestration script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data preprocessing
│   ├── models.py             # Ensemble models
│   ├── explainability.py     # SHAP explainer
│   ├── monitoring.py         # PSI monitoring
│   └── dashboard.py          # Admin dashboard
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_monitoring.py
├── models/                   # Saved models (created at runtime)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── nn_model.h5
│   └── preprocessor.pkl
└── templates/                # Dashboard HTML templates
    └── dashboard.html
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Ensemble weights**: Adjust model contributions
- **PSI thresholds**: Set drift sensitivity
- **SHAP parameters**: Control explanation detail
- **Dashboard settings**: Configure host/port
- **Monitoring intervals**: Set check frequency

## 🔍 Key Components

### ChurnPreprocessor
- Handles missing values
- Encodes categorical features
- Scales numeric features
- Preserves feature names

### EnsembleClassifier
- Trains multiple models
- Weighted probability aggregation
- Individual model access
- Save/load functionality

### SHAPExplainer
- Parallel SHAP computation
- Summary and waterfall plots
- Feature importance ranking
- Model-agnostic approach

### PSIMonitor
- Reference distribution fitting
- PSI calculation per feature
- Drift status determination
- Alert generation

### BackgroundPSIMonitor
- Threaded monitoring
- Periodic drift checks
- History tracking
- Configurable intervals

### AdminDashboard
- Flask web server
- REST API
- Real-time updates
- Alert visualization

## 📈 Model Performance

The ensemble approach combines three complementary models:

- **XGBoost**: Excellent for structured data, handles non-linearity
- **LightGBM**: Fast training, efficient memory usage
- **Neural Network**: Captures complex patterns, flexible architecture

Weighted ensemble typically achieves better performance than individual models.

## 🎯 Use Cases

1. **Proactive Customer Retention**: Identify at-risk customers early
2. **Campaign Targeting**: Focus retention efforts on high-risk segments
3. **Model Monitoring**: Detect when retraining is needed
4. **Regulatory Compliance**: Explain predictions with SHAP
5. **Operations Dashboard**: Monitor system health in production

## 🔒 Security & Privacy

- No sensitive data logged
- Configurable data retention
- Secure API endpoints (add authentication in production)
- Model versioning support

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional model types
- More drift detection methods
- Enhanced dashboard features
- Performance optimizations
- Documentation improvements

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙋 Support

For issues, questions, or suggestions, please open an issue on GitHub.
