"""
Ensemble Models Module for Telecom Churn Prediction
Implements XGBoost, LightGBM, and Neural Network models
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import config


class XGBoostModel:
    """XGBoost classifier wrapper"""
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        return self


class LightGBMModel:
    """LightGBM classifier wrapper"""
    
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        return self


class NeuralNetworkModel:
    """Neural Network classifier wrapper"""
    
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.model = None
        
    def build_model(self, input_dim):
        """Build neural network architecture"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=0):
        """Train the model"""
        if self.model is None:
            self.input_dim = X.shape[1]
            self.model = self.build_model(self.input_dim)
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2
        )
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict(X, verbose=0)
    
    def predict(self, X):
        """Predict classes"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def save(self, filepath):
        """Save model to disk"""
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        return self


class EnsembleClassifier:
    """
    Weighted ensemble of XGBoost, LightGBM, and Neural Network
    """
    
    def __init__(self, weights=None):
        """
        Initialize ensemble classifier
        
        Args:
            weights: Dict with keys 'xgboost', 'lightgbm', 'neural_network'
        """
        self.xgb_model = XGBoostModel()
        self.lgb_model = LightGBMModel()
        self.nn_model = NeuralNetworkModel()
        
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
    def fit(self, X, y):
        """
        Train all models in the ensemble
        
        Args:
            X: Training features
            y: Training labels
        """
        print("Training XGBoost model...")
        self.xgb_model.fit(X, y)
        
        print("Training LightGBM model...")
        self.lgb_model.fit(X, y)
        
        print("Training Neural Network model...")
        self.nn_model.fit(X, y, verbose=0)
        
        print("Ensemble training completed!")
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted ensemble
        
        Args:
            X: Input features
        
        Returns:
            Weighted probability predictions
        """
        # Get predictions from each model
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)
        nn_proba = self.nn_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.weights['xgboost'] * xgb_proba +
            self.weights['lightgbm'] * lgb_proba +
            self.weights['neural_network'] * nn_proba
        )
        
        return ensemble_proba
    
    def predict(self, X):
        """
        Predict classes using weighted ensemble
        
        Args:
            X: Input features
        
        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_model_predictions(self, X):
        """
        Get predictions from individual models
        
        Args:
            X: Input features
        
        Returns:
            Dict with predictions from each model
        """
        return {
            'xgboost': self.xgb_model.predict_proba(X),
            'lightgbm': self.lgb_model.predict_proba(X),
            'neural_network': self.nn_model.predict_proba(X),
            'ensemble': self.predict_proba(X)
        }
    
    def save(self):
        """Save all models"""
        self.xgb_model.save(config.XGBOOST_MODEL_PATH)
        self.lgb_model.save(config.LIGHTGBM_MODEL_PATH)
        self.nn_model.save(config.NN_MODEL_PATH)
        
    def load(self):
        """Load all models"""
        self.xgb_model.load(config.XGBOOST_MODEL_PATH)
        self.lgb_model.load(config.LIGHTGBM_MODEL_PATH)
        self.nn_model.load(config.NN_MODEL_PATH)
        return self
