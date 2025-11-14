"""
Data Preprocessing Module for Telecom Churn Prediction
Handles data cleaning, feature engineering, and transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


class ChurnPreprocessor:
    """Preprocessor for telecom churn data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit the preprocessor on training data
        
        Args:
            X: DataFrame with features
            y: Target variable (optional)
        
        Returns:
            self
        """
        X = X.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit label encoders for categorical columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Fit scaler on all numeric features
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def transform(self, X):
        """
        Transform input data
        
        Args:
            X: DataFrame with features
        
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Transform categorical columns
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        
        Args:
            X: DataFrame with features
            y: Target variable (optional)
        
        Returns:
            Transformed numpy array
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        return X
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath):
        """Load preprocessor from disk"""
        return joblib.load(filepath)


def prepare_sample_data():
    """
    Generate sample telecom churn data for demonstration
    
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample features
    data = {
        'tenure': np.random.randint(1, 73, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target (churn) with some logic
    churn_prob = (
        0.1 + 
        0.3 * (df['contract_type'] == 'Month-to-month').astype(int) +
        0.2 * (df['tenure'] < 12).astype(int) +
        0.15 * (df['internet_service'] == 'Fiber optic').astype(int) +
        0.1 * (df['monthly_charges'] > 80).astype(int)
    )
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df, y
