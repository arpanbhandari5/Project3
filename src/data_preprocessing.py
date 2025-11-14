"""
Data Preprocessing Module for Telecom Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Class to handle data preprocessing for telecom churn prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_copy = df.copy()
        
        if strategy == 'drop':
            df_copy = df_copy.dropna()
        else:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df_copy[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
                        
            # For categorical columns, fill with mode
            cat_cols = df_copy.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_copy[col].isnull().sum() > 0:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                    
        print(f"Missing values handled. New shape: {df_copy.shape}")
        return df_copy
    
    def encode_categorical(self, df, categorical_cols):
        """
        Encode categorical variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_cols (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        for col in categorical_cols:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le
                
        print(f"Categorical encoding completed for {len(categorical_cols)} columns")
        return df_copy
    
    def scale_features(self, X_train, X_test):
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: Scaled training and test features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed. Train size: {X_train.shape}, Test size: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath, target_col, categorical_cols=None, test_size=0.2):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath (str): Path to the data file
            target_col (str): Name of the target column
            categorical_cols (list): List of categorical columns
            test_size (float): Proportion of test set
            
        Returns:
            tuple: Preprocessed train and test data
        """
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        if categorical_cols:
            df = self.encode_categorical(df, categorical_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, target_col, test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("Preprocessing pipeline completed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test
