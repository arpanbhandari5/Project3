"""
Example usage of the Telecom Churn Prediction System
Demonstrates various features and use cases
"""

import pandas as pd
from main import ChurnPredictionPipeline
from src.preprocessing import prepare_sample_data
from sklearn.model_selection import train_test_split


def example_1_train_and_predict():
    """Example 1: Train pipeline and make predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Train Pipeline and Make Predictions")
    print("="*60)
    
    # Generate sample data
    X, y = prepare_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train pipeline
    pipeline = ChurnPredictionPipeline()
    pipeline.train(X_train, y_train, X_test, y_test)
    
    # Make batch predictions
    predictions = pipeline.predict(X_test.head(10))
    probas = pipeline.predict(X_test.head(10), return_proba=True)
    
    print(f"\nFirst 10 predictions: {predictions}")
    print(f"First 5 churn probabilities: {probas[:5, 1]}")


def example_2_single_prediction_with_explanation():
    """Example 2: Single prediction with SHAP explanation"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Prediction with Explanation")
    print("="*60)
    
    # Load trained pipeline
    pipeline = ChurnPredictionPipeline()
    pipeline.load()
    
    # Create a sample customer
    customer = {
        'tenure': 6,
        'monthly_charges': 89.50,
        'total_charges': 537.00,
        'contract_type': 'Month-to-month',
        'internet_service': 'Fiber optic',
        'payment_method': 'Electronic check',
        'gender': 'Female',
        'senior_citizen': 0,
        'partner': 'No',
        'dependents': 'No'
    }
    
    # Get prediction with explanation
    result = pipeline.predict_single(customer, explain=True)
    
    print(f"\nCustomer Profile:")
    print(f"  Tenure: {customer['tenure']} months")
    print(f"  Monthly Charges: ${customer['monthly_charges']}")
    print(f"  Contract: {customer['contract_type']}")
    
    print(f"\nPrediction Results:")
    print(f"  Will Churn: {'YES' if result['prediction'] == 1 else 'NO'}")
    print(f"  Churn Probability: {result['churn_probability']:.2%}")
    
    print(f"\nModel Breakdown:")
    print(f"  XGBoost: {result['model_predictions']['xgboost']:.4f}")
    print(f"  LightGBM: {result['model_predictions']['lightgbm']:.4f}")
    print(f"  Neural Network: {result['model_predictions']['neural_network']:.4f}")
    
    if 'top_features' in result:
        print(f"\nTop Contributing Features:")
        for feat in result['top_features']:
            direction = "increases" if feat['shap_value'] > 0 else "decreases"
            print(f"  • {feat['feature']}: {direction} churn risk (SHAP: {feat['shap_value']:.4f})")


def example_3_drift_monitoring():
    """Example 3: Monitor data drift"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Monitor Data Drift")
    print("="*60)
    
    # Load pipeline
    pipeline = ChurnPredictionPipeline()
    pipeline.load()
    
    # Generate reference and new data
    X_ref, _ = prepare_sample_data()
    X_new, _ = prepare_sample_data()
    
    # Initialize PSI monitor with reference data
    from src.monitoring import PSIMonitor
    X_ref_processed = pipeline.preprocessor.transform(X_ref)
    pipeline.psi_monitor = PSIMonitor(X_ref_processed, feature_names=pipeline.preprocessor.feature_names)
    
    # Check for drift
    drift_status = pipeline.monitor_drift(X_new)
    
    print(f"\nDrift Check Results:")
    print(f"  Overall Status: {drift_status['overall_status'].upper()}")
    print(f"  Timestamp: {drift_status['timestamp']}")
    
    if drift_status['alerts']:
        print(f"\n  ⚠️  Active Alerts ({len(drift_status['alerts'])}):")
        for alert in drift_status['alerts']:
            print(f"    • {alert['feature']}: PSI={alert['psi']:.4f} ({alert['severity']})")
    else:
        print(f"\n  ✓ No alerts - data is stable")
    
    # Show per-feature PSI values
    print(f"\n  Feature PSI Values:")
    for feature, info in drift_status['features'].items():
        status_symbol = "✓" if info['status'] == 'stable' else "⚠"
        print(f"    {status_symbol} {feature}: {info['psi']:.4f} ({info['status']})")


def example_4_batch_processing():
    """Example 4: Batch processing with explanations"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing with Explanations")
    print("="*60)
    
    # Load pipeline
    pipeline = ChurnPredictionPipeline()
    pipeline.load()
    
    # Generate batch data
    X_batch, _ = prepare_sample_data()
    X_batch = X_batch.head(5)  # Process 5 customers
    
    # Get predictions with SHAP explanations
    predictions, shap_values = pipeline.predict(X_batch, explain=True)
    
    print(f"\nProcessed {len(X_batch)} customers:")
    for i, (pred, prob) in enumerate(zip(predictions, pipeline.predict(X_batch, return_proba=True))):
        print(f"  Customer {i+1}: {'CHURN' if pred == 1 else 'RETAIN'} "
              f"(probability: {prob[1]:.2%})")


def example_5_model_comparison():
    """Example 5: Compare individual model predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Individual Model Predictions")
    print("="*60)
    
    # Load pipeline
    pipeline = ChurnPredictionPipeline()
    pipeline.load()
    
    # Generate test data
    X_test, _ = prepare_sample_data()
    X_test = X_test.head(1)
    
    # Preprocess
    X_processed = pipeline.preprocessor.transform(X_test)
    
    # Get predictions from all models
    model_preds = pipeline.ensemble_model.get_model_predictions(X_processed)
    
    print(f"\nModel Comparison for Sample Customer:")
    print(f"  XGBoost:")
    print(f"    No Churn: {model_preds['xgboost'][0][0]:.4f}")
    print(f"    Churn: {model_preds['xgboost'][0][1]:.4f}")
    
    print(f"  LightGBM:")
    print(f"    No Churn: {model_preds['lightgbm'][0][0]:.4f}")
    print(f"    Churn: {model_preds['lightgbm'][0][1]:.4f}")
    
    print(f"  Neural Network:")
    print(f"    No Churn: {model_preds['neural_network'][0][0]:.4f}")
    print(f"    Churn: {model_preds['neural_network'][0][1]:.4f}")
    
    print(f"\n  Ensemble (Weighted):")
    print(f"    No Churn: {model_preds['ensemble'][0][0]:.4f}")
    print(f"    Churn: {model_preds['ensemble'][0][1]:.4f}")
    
    print(f"\n  Ensemble Weights:")
    for model, weight in pipeline.ensemble_model.weights.items():
        print(f"    {model}: {weight:.2%}")


if __name__ == "__main__":
    import sys
    
    # Run all examples or specific one
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_train_and_predict()
        elif example_num == "2":
            example_2_single_prediction_with_explanation()
        elif example_num == "3":
            example_3_drift_monitoring()
        elif example_num == "4":
            example_4_batch_processing()
        elif example_num == "5":
            example_5_model_comparison()
        else:
            print("Usage: python example_usage.py [1-5]")
    else:
        # Run examples 2-5 (skip 1 since models are already trained)
        print("\nRunning all examples...")
        print("(Skipping Example 1 - models already trained from demo)")
        example_2_single_prediction_with_explanation()
        example_3_drift_monitoring()
        example_4_batch_processing()
        example_5_model_comparison()
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60)
