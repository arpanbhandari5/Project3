"""
PSI (Population Stability Index) Monitoring Module
Tracks model drift and triggers alerts when thresholds are exceeded
"""

import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
import config


class PSIMonitor:
    """Monitor data drift using Population Stability Index"""
    
    def __init__(self, reference_data, feature_names=None, n_bins=10):
        """
        Initialize PSI monitor
        
        Args:
            reference_data: Reference dataset (numpy array or DataFrame)
            feature_names: List of feature names
            n_bins: Number of bins for PSI calculation
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        self.n_bins = n_bins
        self.reference_distributions = {}
        self.alerts = []
        self._fit_reference()
        
    def _fit_reference(self):
        """Compute reference distributions for each feature"""
        for i, feature_name in enumerate(self.feature_names):
            feature_data = self.reference_data[:, i]
            
            # Create bins
            _, bin_edges = np.histogram(feature_data, bins=self.n_bins)
            
            # Compute reference distribution
            ref_counts, _ = np.histogram(feature_data, bins=bin_edges)
            ref_dist = ref_counts / len(feature_data)
            
            # Avoid zero probabilities
            ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
            
            self.reference_distributions[feature_name] = {
                'distribution': ref_dist,
                'bin_edges': bin_edges
            }
    
    def calculate_psi(self, current_data, feature_idx=None):
        """
        Calculate PSI for current data
        
        Args:
            current_data: Current dataset (numpy array)
            feature_idx: Specific feature index (None for all features)
        
        Returns:
            Dict with PSI values for each feature
        """
        psi_values = {}
        
        features_to_check = (
            [(feature_idx, self.feature_names[feature_idx])] 
            if feature_idx is not None 
            else enumerate(self.feature_names)
        )
        
        for i, feature_name in features_to_check:
            current_feature = current_data[:, i]
            ref_info = self.reference_distributions[feature_name]
            
            # Compute current distribution using same bins
            current_counts, _ = np.histogram(
                current_feature, 
                bins=ref_info['bin_edges']
            )
            current_dist = current_counts / len(current_feature)
            
            # Avoid zero probabilities
            current_dist = np.where(current_dist == 0, 0.0001, current_dist)
            
            # Calculate PSI
            psi = np.sum(
                (current_dist - ref_info['distribution']) * 
                np.log(current_dist / ref_info['distribution'])
            )
            
            psi_values[feature_name] = psi
        
        return psi_values
    
    def check_drift(self, current_data):
        """
        Check for drift and generate alerts
        
        Args:
            current_data: Current dataset
        
        Returns:
            Dict with drift status and alerts
        """
        psi_values = self.calculate_psi(current_data)
        
        drift_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'stable',
            'features': {},
            'alerts': []
        }
        
        for feature_name, psi_value in psi_values.items():
            status = self._get_drift_status(psi_value)
            drift_status['features'][feature_name] = {
                'psi': float(psi_value),
                'status': status
            }
            
            if status in ['warning', 'critical']:
                alert = {
                    'feature': feature_name,
                    'psi': float(psi_value),
                    'severity': status,
                    'timestamp': drift_status['timestamp']
                }
                drift_status['alerts'].append(alert)
                self.alerts.append(alert)
        
        # Determine overall status
        if any(v['status'] == 'critical' for v in drift_status['features'].values()):
            drift_status['overall_status'] = 'critical'
        elif any(v['status'] == 'warning' for v in drift_status['features'].values()):
            drift_status['overall_status'] = 'warning'
        
        return drift_status
    
    def _get_drift_status(self, psi_value):
        """
        Determine drift status based on PSI value
        
        Args:
            psi_value: PSI value
        
        Returns:
            Status string: 'stable', 'warning', or 'critical'
        """
        if psi_value < config.PSI_THRESHOLD_WARNING:
            return 'stable'
        elif psi_value < config.PSI_THRESHOLD_CRITICAL:
            return 'warning'
        else:
            return 'critical'
    
    def get_alerts(self, severity=None):
        """
        Get recorded alerts
        
        Args:
            severity: Filter by severity ('warning', 'critical', or None for all)
        
        Returns:
            List of alerts
        """
        if severity:
            return [a for a in self.alerts if a['severity'] == severity]
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []


class BackgroundPSIMonitor:
    """Background monitoring service with periodic checks"""
    
    def __init__(self, psi_monitor, check_interval=None):
        """
        Initialize background monitor
        
        Args:
            psi_monitor: PSIMonitor instance
            check_interval: Check interval in seconds
        """
        self.psi_monitor = psi_monitor
        self.check_interval = check_interval or config.PSI_CHECK_INTERVAL
        self.monitoring_thread = None
        self.is_running = False
        self.drift_history = []
        
    def start(self, data_source_func):
        """
        Start background monitoring
        
        Args:
            data_source_func: Callable that returns current data for monitoring
        """
        if self.is_running:
            print("Background monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(data_source_func,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Background PSI monitoring started (interval: {self.check_interval}s)")
    
    def _monitoring_loop(self, data_source_func):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get current data
                current_data = data_source_func()
                
                if current_data is not None and len(current_data) > 0:
                    # Check for drift
                    drift_status = self.psi_monitor.check_drift(current_data)
                    self.drift_history.append(drift_status)
                    
                    # Log alerts
                    if drift_status['alerts']:
                        for alert in drift_status['alerts']:
                            print(f"[DRIFT ALERT] {alert['severity'].upper()}: "
                                  f"{alert['feature']} (PSI: {alert['psi']:.4f})")
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def stop(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Background PSI monitoring stopped")
    
    def get_drift_history(self, limit=None):
        """
        Get drift check history
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of drift status records
        """
        if limit:
            return self.drift_history[-limit:]
        return self.drift_history
