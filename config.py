"""
Configuration file for Telecom Churn Prediction Project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Data paths
DEFAULT_DATA_PATH = os.path.join(RAW_DATA_DIR, 'telecom_churn_data.csv')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_churn_model.pkl')

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target column
TARGET_COLUMN = 'Churn'

# Categorical columns (excluding customerID and target)
CATEGORICAL_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Numerical columns
NUMERICAL_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Model hyperparameters (can be customized)
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# Visualization settings
FIGURE_SIZE = (12, 8)
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl'

# Data generation settings
DEFAULT_SAMPLE_SIZE = 5000
CHURN_RATE_TARGET = 0.27  # Target churn rate for synthetic data
