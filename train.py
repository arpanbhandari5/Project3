"""
Main Training Script for Telecom Churn Prediction
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ChurnModelTrainer
from visualization import ChurnVisualizer
import pandas as pd
import numpy as np


def main():
    """
    Main function to run the complete training pipeline
    """
    print("=" * 80)
    print("TELECOM CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 80)
    
    # Configuration
    DATA_PATH = "data/raw/telecom_churn_data.csv"
    MODEL_SAVE_PATH = "models/best_churn_model.pkl"
    TARGET_COL = "Churn"
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nData file not found at {DATA_PATH}")
        print("Generating sample data...")
        from data_generator import generate_sample_telecom_data
        generate_sample_telecom_data(n_samples=5000, output_path=DATA_PATH)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ChurnModelTrainer()
    visualizer = ChurnVisualizer()
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 80)
    
    # Load data
    df = preprocessor.load_data(DATA_PATH)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nChurn Distribution:")
    print(df[TARGET_COL].value_counts())
    print(f"\nChurn Rate: {(df[TARGET_COL].value_counts()['Yes'] / len(df) * 100):.2f}%")
    
    # Identify categorical columns (excluding target and customerID)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
        df = df.drop('customerID', axis=1)
    
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)
    
    print(f"\nCategorical columns to encode: {categorical_cols}")
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Convert target to binary
    if df[TARGET_COL].dtype == 'object':
        df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    df = preprocessor.encode_categorical(df, categorical_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, TARGET_COL, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Initialize and train all models
    trainer.initialize_models()
    results_df = trainer.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Get detailed report for best model
    print("\n" + "=" * 80)
    print(f"DETAILED EVALUATION - BEST MODEL: {trainer.best_model_name}")
    print("=" * 80)
    
    trainer.get_classification_report(X_test_scaled, y_test)
    trainer.get_confusion_matrix(X_test_scaled, y_test)
    
    print("\n" + "=" * 80)
    print("STEP 3: SAVING MODEL")
    print("=" * 80)
    
    # Save the best model
    trainer.save_model(MODEL_SAVE_PATH)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    print(f"F1-Score: {trainer.results[trainer.best_model_name]['metrics']['f1_score']:.4f}")
    print(f"Accuracy: {trainer.results[trainer.best_model_name]['metrics']['accuracy']:.4f}")
    
    # Save results
    results_path = "models/model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nModel comparison results saved at: {results_path}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
