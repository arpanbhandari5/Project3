"""
Tests for preprocessing module
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import ChurnPreprocessor, prepare_sample_data


class TestChurnPreprocessor(unittest.TestCase):
    """Test ChurnPreprocessor class"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = prepare_sample_data()
        self.preprocessor = ChurnPreprocessor()
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        X_transformed = self.preprocessor.fit_transform(self.X)
        
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(X_transformed.shape[0], len(self.X))
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_transform_after_fit(self):
        """Test transform method after fitting"""
        self.preprocessor.fit(self.X)
        X_transformed = self.preprocessor.transform(self.X)
        
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(X_transformed.shape[0], len(self.X))
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error"""
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.X)
    
    def test_prepare_sample_data(self):
        """Test sample data generation"""
        X, y = prepare_sample_data()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(len(X), len(y))
        self.assertTrue(len(X) > 0)


class TestPrepareData(unittest.TestCase):
    """Test data preparation functions"""
    
    def test_sample_data_structure(self):
        """Test structure of sample data"""
        X, y = prepare_sample_data()
        
        # Check required columns
        required_cols = ['tenure', 'monthly_charges', 'total_charges']
        for col in required_cols:
            self.assertIn(col, X.columns)
        
        # Check target values
        unique_y = np.unique(y)
        self.assertEqual(len(unique_y), 2)
        self.assertIn(0, unique_y)
        self.assertIn(1, unique_y)


if __name__ == '__main__':
    unittest.main()
