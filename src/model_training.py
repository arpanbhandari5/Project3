"""
Model Training Module for Telecom Churn Prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class ChurnModelTrainer:
    """
    Class to train and evaluate churn prediction models
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """
        Initialize different models for comparison
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42, probability=True)
        }
        print(f"Initialized {len(self.models)} models")
        
    def train_model(self, model, X_train, y_train):
        """
        Train a single model
        
        Args:
            model: Machine learning model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Results dataframe with all model metrics
        """
        if not self.models:
            self.initialize_models()
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            trained_model = self.train_model(model, X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test)
            metrics['model_name'] = name
            
            results_list.append(metrics)
            self.results[name] = {'model': trained_model, 'metrics': metrics}
            
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Identify best model
        self.best_model_name = results_df.iloc[0]['model_name']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        
        return results_df
    
    def get_classification_report(self, X_test, y_test, model_name=None):
        """
        Get detailed classification report
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model (uses best model if None)
            
        Returns:
            str: Classification report
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.results[model_name]['model']
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        print(f"\nClassification Report for {model_name}:")
        print(report)
        
        return report
    
    def get_confusion_matrix(self, X_test, y_test, model_name=None):
        """
        Get confusion matrix
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model (uses best model if None)
            
        Returns:
            np.array: Confusion matrix
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.results[model_name]['model']
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nConfusion Matrix for {model_name}:")
        print(cm)
        
        return cm
    
    def save_model(self, model_path, model_name=None):
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save the model
            model_name: Name of the model to save (uses best model if None)
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.results[model_name]['model']
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load trained model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained model
        
        Args:
            X: Features to predict
            model_name: Name of the model (uses best model if None)
            
        Returns:
            np.array: Predictions
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.results[model_name]['model']
        
        predictions = model.predict(X)
        return predictions
    
    def predict_proba(self, X, model_name=None):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            model_name: Name of the model (uses best model if None)
            
        Returns:
            np.array: Prediction probabilities
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.results[model_name]['model']
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            return probabilities
        else:
            print("Model does not support probability predictions")
            return None
