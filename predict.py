"""
Prediction Script for Telecom Churn Prediction
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ChurnModelTrainer
import pandas as pd
import numpy as np


def predict_churn(data_path, model_path):
    """
    Make churn predictions on new data
    
    Args:
        data_path: Path to the input data CSV file
        model_path: Path to the trained model
    """
    print("=" * 80)
    print("TELECOM CHURN PREDICTION - INFERENCE")
    print("=" * 80)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ChurnModelTrainer()
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = trainer.load_model(model_path)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Keep customer IDs for reference
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].values
        df_processed = df.drop('customerID', axis=1)
    else:
        customer_ids = np.arange(len(df))
        df_processed = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Handle missing values
    df_processed = preprocessor.handle_missing_values(df_processed, strategy='mean')
    
    # Encode categorical variables
    if categorical_cols:
        df_processed = preprocessor.encode_categorical(df_processed, categorical_cols)
    
    # Scale features
    X_scaled = preprocessor.scaler.fit_transform(df_processed)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_scaled)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        probabilities = None
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'customerID': customer_ids,
        'Churn_Prediction': ['Yes' if p == 1 else 'No' for p in predictions]
    })
    
    if probabilities is not None:
        results_df['Churn_Probability'] = np.round(probabilities * 100, 2)
    
    # Save results
    output_path = data_path.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\nTotal customers: {len(predictions)}")
    print(f"Predicted to churn: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.2f}%)")
    print(f"Predicted to stay: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.2f}%)")
    
    print(f"\nPredictions saved to: {output_path}")
    
    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    
    return results_df


def main():
    """
    Main function for prediction
    """
    parser = argparse.ArgumentParser(description='Predict telecom customer churn')
    parser.add_argument('--data', type=str, default='data/raw/telecom_churn_data.csv',
                        help='Path to input data CSV file')
    parser.add_argument('--model', type=str, default='models/best_churn_model.pkl',
                        help='Path to trained model file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please run train.py first to train a model.")
        return
    
    # Make predictions
    predict_churn(args.data, args.model)


if __name__ == "__main__":
    main()
