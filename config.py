"""
Configuration file for Telecom Churn Prediction System
"""

# Model ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.4,
    'lightgbm': 0.35,
    'neural_network': 0.25
}

# PSI drift thresholds
PSI_THRESHOLD_WARNING = 0.1
PSI_THRESHOLD_CRITICAL = 0.25

# Model paths
MODEL_DIR = 'models'
XGBOOST_MODEL_PATH = f'{MODEL_DIR}/xgboost_model.pkl'
LIGHTGBM_MODEL_PATH = f'{MODEL_DIR}/lightgbm_model.pkl'
NN_MODEL_PATH = f'{MODEL_DIR}/nn_model.h5'
PREPROCESSOR_PATH = f'{MODEL_DIR}/preprocessor.pkl'

# SHAP configuration
SHAP_SAMPLE_SIZE = 100
SHAP_MAX_DISPLAY = 20

# Dashboard configuration
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = True

# Monitoring configuration
PSI_CHECK_INTERVAL = 3600  # seconds
PSI_REFERENCE_DATA_PATH = 'data/reference_data.csv'
