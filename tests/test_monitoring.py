"""
Tests for PSI monitoring module
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import prepare_sample_data, ChurnPreprocessor
from monitoring import PSIMonitor, BackgroundPSIMonitor


class TestPSIMonitor(unittest.TestCase):
    """Test PSIMonitor class"""
    
    def setUp(self):
        """Set up test data"""
        X, y = prepare_sample_data()
        preprocessor = ChurnPreprocessor()
        self.X = preprocessor.fit_transform(X)
        
        # Reference and current data
        self.X_ref = self.X[:500]
        self.X_current = self.X[500:600]
        
        self.monitor = PSIMonitor(self.X_ref)
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor.reference_distributions)
        self.assertEqual(len(self.monitor.reference_distributions), self.X_ref.shape[1])
    
    def test_calculate_psi(self):
        """Test PSI calculation"""
        psi_values = self.monitor.calculate_psi(self.X_current)
        
        self.assertIsInstance(psi_values, dict)
        self.assertEqual(len(psi_values), self.X_ref.shape[1])
        
        # PSI values should be non-negative
        for psi in psi_values.values():
            self.assertGreaterEqual(psi, 0)
    
    def test_check_drift(self):
        """Test drift checking"""
        drift_status = self.monitor.check_drift(self.X_current)
        
        self.assertIn('timestamp', drift_status)
        self.assertIn('overall_status', drift_status)
        self.assertIn('features', drift_status)
        self.assertIn('alerts', drift_status)
        
        self.assertIn(drift_status['overall_status'], ['stable', 'warning', 'critical'])
    
    def test_drift_with_shifted_data(self):
        """Test drift detection with artificially shifted data"""
        # Create shifted data
        X_shifted = self.X_current.copy()
        X_shifted[:, 0] = X_shifted[:, 0] + 2.0  # Shift first feature
        
        drift_status = self.monitor.check_drift(X_shifted)
        
        # Should detect some drift
        feature_0_status = drift_status['features']['feature_0']['status']
        self.assertIn(feature_0_status, ['warning', 'critical'])
    
    def test_get_alerts(self):
        """Test alert retrieval"""
        # Generate some alerts by checking drift
        X_shifted = self.X_current.copy()
        X_shifted[:, 0] = X_shifted[:, 0] + 3.0
        self.monitor.check_drift(X_shifted)
        
        alerts = self.monitor.get_alerts()
        self.assertIsInstance(alerts, list)
    
    def test_clear_alerts(self):
        """Test clearing alerts"""
        # Generate alerts
        X_shifted = self.X_current.copy()
        X_shifted[:, 0] = X_shifted[:, 0] + 3.0
        self.monitor.check_drift(X_shifted)
        
        # Clear and verify
        self.monitor.clear_alerts()
        alerts = self.monitor.get_alerts()
        self.assertEqual(len(alerts), 0)


class TestBackgroundPSIMonitor(unittest.TestCase):
    """Test BackgroundPSIMonitor class"""
    
    def setUp(self):
        """Set up test data"""
        X, y = prepare_sample_data()
        preprocessor = ChurnPreprocessor()
        self.X = preprocessor.fit_transform(X)
        
        self.X_ref = self.X[:500]
        self.X_current = self.X[500:600]
        
        psi_monitor = PSIMonitor(self.X_ref)
        self.bg_monitor = BackgroundPSIMonitor(psi_monitor, check_interval=1)
    
    def test_initialization(self):
        """Test background monitor initialization"""
        self.assertIsNotNone(self.bg_monitor.psi_monitor)
        self.assertFalse(self.bg_monitor.is_running)
        self.assertEqual(self.bg_monitor.check_interval, 1)
    
    def test_start_stop(self):
        """Test starting and stopping monitoring"""
        data_source = lambda: self.X_current
        
        self.bg_monitor.start(data_source)
        self.assertTrue(self.bg_monitor.is_running)
        
        # Wait for at least one check
        time.sleep(1.5)
        
        self.bg_monitor.stop()
        self.assertFalse(self.bg_monitor.is_running)
    
    def test_drift_history(self):
        """Test drift history recording"""
        data_source = lambda: self.X_current
        
        self.bg_monitor.start(data_source)
        time.sleep(1.5)  # Wait for checks
        self.bg_monitor.stop()
        
        history = self.bg_monitor.get_drift_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)


if __name__ == '__main__':
    unittest.main()
