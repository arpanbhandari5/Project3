"""
Main orchestration script for Telecom Churn Prediction System
Implements end-to-end flow: input → preprocessing → ensemble → classification
Includes parallel SHAP explainability and PSI monitoring
"""

import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import config

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import ChurnPreprocessor, prepare_sample_data
from models import EnsembleClassifier
from explainability import SHAPExplainer
from monitoring import PSIMonitor, BackgroundPSIMonitor
from dashboard import AdminDashboard, create_dashboard_template


class ChurnPredictionPipeline:
    """End-to-end churn prediction pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.ensemble_model = None
        self.shap_explainer = None
        self.psi_monitor = None
        self.background_monitor = None
        self.dashboard = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the complete pipeline
        
        Args:
            X_train: Training features (DataFrame)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("=" * 60)
        print("TRAINING CHURN PREDICTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Preprocessing
        print("\n[1/4] Preprocessing data...")
        self.preprocessor = ChurnPreprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        print(f"✓ Data preprocessed: {X_train_processed.shape}")
        
        # Step 2: Train ensemble models
        print("\n[2/4] Training ensemble models...")
        self.ensemble_model = EnsembleClassifier()
        self.ensemble_model.fit(X_train_processed, y_train)
        print("✓ Ensemble models trained")
        
        # Step 3: Initialize SHAP explainer (parallel)
        print("\n[3/4] Initializing SHAP explainer...")
        self.shap_explainer = SHAPExplainer(
            self.ensemble_model,
            X_train_processed,
            feature_names=self.preprocessor.feature_names
        )
        self.shap_explainer.fit()
        print("✓ SHAP explainer initialized")
        
        # Step 4: Initialize PSI monitor
        print("\n[4/4] Setting up PSI monitoring...")
        self.psi_monitor = PSIMonitor(
            X_train_processed,
            feature_names=self.preprocessor.feature_names
        )
        print("✓ PSI monitor initialized")
        
        # Validation
        if X_val is not None and y_val is not None:
            print("\n" + "=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            X_val_processed = self.preprocessor.transform(X_val)
            predictions = self.ensemble_model.predict(X_val_processed)
            accuracy = (predictions == y_val).mean()
            print(f"Validation Accuracy: {accuracy:.4f}")
        
        print("\n✓ Pipeline training completed!")
        return self
    
    def predict(self, X, return_proba=False, explain=False):
        """
        Make predictions on new data
        
        Args:
            X: Input features (DataFrame)
            return_proba: Return probabilities instead of classes
            explain: Generate SHAP explanations
        
        Returns:
            Predictions (and optionally SHAP values)
        """
        if self.preprocessor is None or self.ensemble_model is None:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # Predict
        if return_proba:
            predictions = self.ensemble_model.predict_proba(X_processed)
        else:
            predictions = self.ensemble_model.predict(X_processed)
        
        # Explain (parallel)
        shap_values = None
        if explain and self.shap_explainer is not None:
            print("Computing SHAP explanations (parallel)...")
            shap_values = self.shap_explainer.explain(X_processed, parallel=True)
            print("✓ SHAP explanations computed")
        
        if explain:
            return predictions, shap_values
        return predictions
    
    def predict_single(self, user_input, explain=True):
        """
        Predict for a single user input
        
        Args:
            user_input: Dictionary or DataFrame with user features
            explain: Generate explanation
        
        Returns:
            Dict with prediction results
        """
        import pandas as pd
        
        # Convert to DataFrame if dict
        if isinstance(user_input, dict):
            user_input = pd.DataFrame([user_input])
        
        # Preprocess
        X_processed = self.preprocessor.transform(user_input)
        
        # Get predictions from all models
        model_predictions = self.ensemble_model.get_model_predictions(X_processed)
        
        # Final prediction
        final_prediction = self.ensemble_model.predict(X_processed)[0]
        final_proba = self.ensemble_model.predict_proba(X_processed)[0]
        
        result = {
            'prediction': int(final_prediction),
            'churn_probability': float(final_proba[1]),
            'model_predictions': {
                'xgboost': float(model_predictions['xgboost'][0][1]),
                'lightgbm': float(model_predictions['lightgbm'][0][1]),
                'neural_network': float(model_predictions['neural_network'][0][1])
            },
            'ensemble_weights': config.ENSEMBLE_WEIGHTS
        }
        
        # Add SHAP explanation
        if explain and self.shap_explainer is not None:
            shap_values = self.shap_explainer.explain(X_processed, parallel=False)
            
            # Get feature importance
            if isinstance(shap_values, list):
                values = shap_values[1][0]  # Churn class
            else:
                values = shap_values[0]
            
            # Ensure values is 1D array
            if values.ndim > 1:
                values = values.flatten()
            
            # Top features
            feature_importance = []
            for fname, val in zip(self.preprocessor.feature_names, values):
                feature_importance.append((fname, float(val)))
            
            sorted_features = sorted(
                feature_importance,
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            result['top_features'] = [
                {'feature': f, 'shap_value': v}
                for f, v in sorted_features
            ]
        
        return result
    
    def monitor_drift(self, X_current):
        """
        Check for data drift
        
        Args:
            X_current: Current data (DataFrame)
        
        Returns:
            Drift status
        """
        if self.psi_monitor is None:
            raise ValueError("PSI monitor not initialized. Train the pipeline first.")
        
        X_processed = self.preprocessor.transform(X_current)
        return self.psi_monitor.check_drift(X_processed)
    
    def start_background_monitoring(self, data_source_func, check_interval=None):
        """
        Start background PSI monitoring
        
        Args:
            data_source_func: Function that returns current data
            check_interval: Check interval in seconds
        """
        if self.psi_monitor is None:
            raise ValueError("PSI monitor not initialized. Train the pipeline first.")
        
        self.background_monitor = BackgroundPSIMonitor(
            self.psi_monitor,
            check_interval=check_interval
        )
        self.background_monitor.start(data_source_func)
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        if self.background_monitor:
            self.background_monitor.stop()
    
    def launch_dashboard(self, host=None, port=None):
        """
        Launch admin dashboard
        
        Args:
            host: Dashboard host
            port: Dashboard port
        """
        # Use background monitor if available, otherwise regular monitor
        monitor = self.background_monitor or self.psi_monitor
        
        if monitor is None:
            raise ValueError("No monitor available. Train the pipeline first.")
        
        # Create dashboard template
        create_dashboard_template()
        
        # Create and run dashboard
        self.dashboard = AdminDashboard(monitor, self.ensemble_model)
        self.dashboard.run(host=host, port=port)
    
    def save(self):
        """Save all pipeline components"""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        print("Saving pipeline components...")
        self.preprocessor.save(config.PREPROCESSOR_PATH)
        self.ensemble_model.save()
        print(f"✓ Pipeline saved to {config.MODEL_DIR}/")
    
    def load(self):
        """Load all pipeline components"""
        print("Loading pipeline components...")
        self.preprocessor = ChurnPreprocessor.load(config.PREPROCESSOR_PATH)
        self.ensemble_model = EnsembleClassifier()
        self.ensemble_model.load()
        print("✓ Pipeline loaded")
        return self


def demo_end_to_end_flow():
    """Demonstrate the complete end-to-end flow"""
    print("\n" + "=" * 60)
    print("TELECOM CHURN PREDICTION - END-TO-END DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    print("\n[Step 1] Generating sample data...")
    X, y = prepare_sample_data()
    print(f"✓ Generated {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train pipeline
    print("\n[Step 2] Initializing and training pipeline...")
    pipeline = ChurnPredictionPipeline()
    pipeline.train(X_train, y_train, X_test, y_test)
    
    # Save pipeline
    print("\n[Step 3] Saving pipeline...")
    pipeline.save()
    
    # Make predictions with explanation
    print("\n[Step 4] Making predictions with SHAP explanations...")
    sample_input = X_test.iloc[0:1]
    result = pipeline.predict_single(sample_input, explain=True)
    
    print("\nPrediction Result:")
    print(f"  Churn Prediction: {'YES' if result['prediction'] == 1 else 'NO'}")
    print(f"  Churn Probability: {result['churn_probability']:.4f}")
    print(f"\n  Individual Model Predictions (churn probability):")
    print(f"    XGBoost: {result['model_predictions']['xgboost']:.4f}")
    print(f"    LightGBM: {result['model_predictions']['lightgbm']:.4f}")
    print(f"    Neural Network: {result['model_predictions']['neural_network']:.4f}")
    
    if 'top_features' in result:
        print(f"\n  Top Contributing Features:")
        for feat in result['top_features']:
            print(f"    {feat['feature']}: {feat['shap_value']:.4f}")
    
    # Check drift
    print("\n[Step 5] Checking for data drift...")
    drift_status = pipeline.monitor_drift(X_test)
    print(f"  Overall Drift Status: {drift_status['overall_status'].upper()}")
    if drift_status['alerts']:
        print(f"  Active Alerts: {len(drift_status['alerts'])}")
        for alert in drift_status['alerts']:
            print(f"    - {alert['feature']}: PSI={alert['psi']:.4f} ({alert['severity']})")
    else:
        print("  ✓ No drift alerts - system is stable")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start background monitoring with: pipeline.start_background_monitoring(data_source_func)")
    print("2. Launch dashboard with: pipeline.launch_dashboard()")
    print("3. Access dashboard at: http://localhost:5000")
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Telecom Churn Prediction System")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    
    args = parser.parse_args()
    
    if args.demo:
        pipeline = demo_end_to_end_flow()
    elif args.dashboard:
        # Load existing pipeline and launch dashboard
        pipeline = ChurnPredictionPipeline()
        pipeline.load()
        pipeline.launch_dashboard(port=args.port)
    else:
        print("Usage:")
        print("  python main.py --demo          # Run demonstration")
        print("  python main.py --dashboard     # Launch dashboard")
