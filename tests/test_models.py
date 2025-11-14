"""
Tests for ensemble models module
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import prepare_sample_data, ChurnPreprocessor
from models import XGBoostModel, LightGBMModel, NeuralNetworkModel, EnsembleClassifier


class TestIndividualModels(unittest.TestCase):
    """Test individual model classes"""
    
    def setUp(self):
        """Set up test data"""
        X, y = prepare_sample_data()
        preprocessor = ChurnPreprocessor()
        self.X = preprocessor.fit_transform(X)
        self.y = y
        
        # Use small subset for faster tests
        self.X_train = self.X[:100]
        self.y_train = y[:100]
        self.X_test = self.X[100:120]
        self.y_test = y[100:120]
    
    def test_xgboost_model(self):
        """Test XGBoost model"""
        model = XGBoostModel()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        proba = model.predict_proba(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)
    
    def test_lightgbm_model(self):
        """Test LightGBM model"""
        model = LightGBMModel()
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        proba = model.predict_proba(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)
    
    def test_neural_network_model(self):
        """Test Neural Network model"""
        model = NeuralNetworkModel()
        model.fit(self.X_train, self.y_train, epochs=5, verbose=0)
        
        predictions = model.predict(self.X_test)
        proba = model.predict_proba(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)


class TestEnsembleClassifier(unittest.TestCase):
    """Test EnsembleClassifier class"""
    
    def setUp(self):
        """Set up test data"""
        X, y = prepare_sample_data()
        preprocessor = ChurnPreprocessor()
        self.X = preprocessor.fit_transform(X)
        self.y = y
        
        # Use small subset for faster tests
        self.X_train = self.X[:100]
        self.y_train = y[:100]
        self.X_test = self.X[100:120]
        self.y_test = y[100:120]
    
    def test_ensemble_training(self):
        """Test ensemble training"""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.X_train, self.y_train)
        
        # Check that all models are trained
        self.assertIsNotNone(ensemble.xgb_model.model)
        self.assertIsNotNone(ensemble.lgb_model.model)
        self.assertIsNotNone(ensemble.nn_model.model)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.X_train, self.y_train)
        
        predictions = ensemble.predict(self.X_test)
        proba = ensemble.predict_proba(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(proba.shape[0], len(self.X_test))
        self.assertEqual(proba.shape[1], 2)
        
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(
            proba.sum(axis=1),
            np.ones(len(self.X_test)),
            decimal=5
        )
    
    def test_get_model_predictions(self):
        """Test getting individual model predictions"""
        ensemble = EnsembleClassifier()
        ensemble.fit(self.X_train, self.y_train)
        
        model_preds = ensemble.get_model_predictions(self.X_test)
        
        self.assertIn('xgboost', model_preds)
        self.assertIn('lightgbm', model_preds)
        self.assertIn('neural_network', model_preds)
        self.assertIn('ensemble', model_preds)
        
        # Check shapes
        for key in model_preds:
            self.assertEqual(model_preds[key].shape[0], len(self.X_test))
    
    def test_custom_weights(self):
        """Test ensemble with custom weights"""
        custom_weights = {
            'xgboost': 0.5,
            'lightgbm': 0.3,
            'neural_network': 0.2
        }
        ensemble = EnsembleClassifier(weights=custom_weights)
        
        # Check weights are normalized
        total = sum(ensemble.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
