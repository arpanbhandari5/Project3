# Quick Start Guide

Get up and running with the Telecom Churn Prediction System in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/arpanbhandari5/Project3.git
cd Project3

# Install dependencies
pip install -r requirements.txt
```

## Run Demo

The fastest way to see the system in action:

```bash
python main.py --demo
```

This will:
- ✓ Generate sample telecom data
- ✓ Train all three models (XGBoost, LightGBM, Neural Network)
- ✓ Initialize SHAP explainer for interpretability
- ✓ Set up PSI monitoring for drift detection
- ✓ Make sample predictions with explanations
- ✓ Check for data drift
- ✓ Save trained models to `models/` directory

**Expected output:**
```
============================================================
TELECOM CHURN PREDICTION - END-TO-END DEMONSTRATION
============================================================

[Step 1] Generating sample data...
✓ Generated 1000 samples with 10 features

[Step 2] Initializing and training pipeline...
============================================================
TRAINING CHURN PREDICTION PIPELINE
============================================================

[1/4] Preprocessing data...
✓ Data preprocessed: (800, 10)

[2/4] Training ensemble models...
Training XGBoost model...
Training LightGBM model...
Training Neural Network model...
Ensemble training completed!
✓ Ensemble models trained

[3/4] Initializing SHAP explainer...
✓ SHAP explainer initialized

[4/4] Setting up PSI monitoring...
✓ PSI monitor initialized

============================================================
VALIDATION RESULTS
============================================================
Validation Accuracy: 0.6550

✓ Pipeline training completed!

[Step 3] Saving pipeline...
✓ Pipeline saved to models/

[Step 4] Making predictions with SHAP explanations...

Prediction Result:
  Churn Prediction: NO
  Churn Probability: 0.4306

  Individual Model Predictions (churn probability):
    XGBoost: 0.3119
    LightGBM: 0.4886
    Neural Network: 0.5391

  Top Contributing Features:
    senior_citizen: 0.2190
    gender: -0.2190
    partner: 0.0669
    dependents: -0.0669
    payment_method: -0.0499

[Step 5] Checking for data drift...
  Overall Drift Status: STABLE
  ✓ No drift alerts - system is stable

============================================================
DEMONSTRATION COMPLETED SUCCESSFULLY!
============================================================
```

## Run Examples

Try different use cases:

```bash
# Run all examples
python example_usage.py

# Or run specific examples
python example_usage.py 2  # Single prediction with explanation
python example_usage.py 3  # Drift monitoring
python example_usage.py 4  # Batch processing
python example_usage.py 5  # Model comparison
```

## Launch Dashboard

After running the demo, start the web dashboard:

```bash
python main.py --dashboard
```

Then open your browser to: **http://localhost:5000**

The dashboard shows:
- ✓ System status and metrics
- ✓ Current drift status
- ✓ Active alerts
- ✓ Real-time updates every 30 seconds

## Make Predictions

### Python API

```python
from main import ChurnPredictionPipeline

# Load trained pipeline
pipeline = ChurnPredictionPipeline()
pipeline.load()

# Make prediction
customer = {
    'tenure': 12,
    'monthly_charges': 75.50,
    'contract_type': 'Month-to-month',
    # ... other features
}

result = pipeline.predict_single(customer, explain=True)
print(f"Churn Probability: {result['churn_probability']:.2%}")
```

### REST API

With dashboard running:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [12, 75.5, 900, 0, 1, 0, 1, 0, 1, 0]}'
```

## Run Tests

Verify everything works:

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test module
python -m unittest tests.test_models -v
```

## Next Steps

1. **Train with your data**: Replace `prepare_sample_data()` with your actual data
2. **Customize models**: Edit `config.py` to adjust ensemble weights
3. **Set up monitoring**: Configure PSI thresholds for your use case
4. **Deploy dashboard**: Use production WSGI server (gunicorn, uWSGI)
5. **Add authentication**: Secure dashboard endpoints for production

## Project Structure

```
Project3/
├── main.py                    # Main orchestration (start here!)
├── example_usage.py           # Usage examples
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── src/
│   ├── preprocessing.py       # Data preprocessing
│   ├── models.py              # Ensemble models
│   ├── explainability.py      # SHAP explainer
│   ├── monitoring.py          # PSI monitoring
│   └── dashboard.py           # Admin dashboard
├── tests/                     # Test suite
└── models/                    # Saved models (created after training)
```

## Troubleshooting

**Issue**: ImportError for tensorflow/xgboost/lightgbm
- **Solution**: `pip install -r requirements.txt`

**Issue**: Models not found
- **Solution**: Run `python main.py --demo` first to train and save models

**Issue**: Dashboard not accessible
- **Solution**: Check firewall settings, try `http://127.0.0.1:5000`

**Issue**: CUDA warnings
- **Solution**: Ignore - system works fine on CPU

## Resources

- **Full Documentation**: See [README.md](README.md)
- **API Reference**: See docstrings in source files
- **Configuration**: Edit [config.py](config.py)

## Support

For issues or questions, open an issue on GitHub.

---

**You're all set!** 🚀 Start with `python main.py --demo` and explore from there.
