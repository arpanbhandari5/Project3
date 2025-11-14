"""
Visualization Module for Telecom Churn Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ChurnVisualizer:
    """
    Class to create visualizations for churn analysis
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with style
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_feature_distribution(self, df, feature, target_col='Churn', figsize=(10, 6)):
        """
        Plot distribution of a feature by churn status
        
        Args:
            df: Dataframe containing the data
            feature: Feature name to plot
            target_col: Target column name
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if df[feature].dtype in ['int64', 'float64']:
            # For numerical features
            df[df[target_col] == 0][feature].hist(alpha=0.5, label='No Churn', bins=30)
            df[df[target_col] == 1][feature].hist(alpha=0.5, label='Churn', bins=30)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.title(f'Distribution of {feature} by Churn Status')
        else:
            # For categorical features
            churn_counts = df.groupby([feature, target_col]).size().unstack(fill_value=0)
            churn_counts.plot(kind='bar', stacked=False)
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.title(f'{feature} by Churn Status')
            plt.legend(['No Churn', 'Churn'])
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df, figsize=(12, 10)):
        """
        Plot correlation matrix heatmap
        
        Args:
            df: Dataframe containing numerical features
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        correlation_matrix = numerical_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_churn_distribution(self, df, target_col='Churn', figsize=(8, 6)):
        """
        Plot churn distribution
        
        Args:
            df: Dataframe containing the data
            target_col: Target column name
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        churn_counts = df[target_col].value_counts()
        
        plt.subplot(1, 2, 1)
        plt.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
                startangle=90, colors=['#66b3ff', '#ff6666'])
        plt.title('Churn Distribution (Pie Chart)')
        
        plt.subplot(1, 2, 2)
        churn_counts.plot(kind='bar', color=['#66b3ff', '#ff6666'])
        plt.xlabel('Churn Status')
        plt.ylabel('Count')
        plt.title('Churn Distribution (Bar Chart)')
        plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, figsize=(8, 6)):
        """
        Plot confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba, figsize=(8, 6)):
        """
        Plot ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            figsize: Figure size
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        plt.figure(figsize=figsize)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results_df, metric='f1_score', figsize=(12, 6)):
        """
        Plot comparison of different models
        
        Args:
            results_df: Dataframe with model results
            metric: Metric to compare
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        results_sorted = results_df.sort_values(metric, ascending=True)
        
        plt.barh(results_sorted['model_name'], results_sorted[metric], color='teal')
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Comparison by {metric.replace("_", " ").title()}', 
                  fontsize=14, fontweight='bold')
        plt.xlim([0, 1])
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_metrics(self, results_df, figsize=(14, 8)):
        """
        Plot multiple metrics for all models
        
        Args:
            results_df: Dataframe with model results
            figsize: Figure size
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("No metrics available to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for idx, metric in enumerate(available_metrics):
            results_sorted = results_df.sort_values(metric, ascending=True)
            axes[idx].barh(results_sorted['model_name'], results_sorted[metric], 
                          color=sns.color_palette("husl", len(results_sorted)))
            axes[idx].set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            axes[idx].set_ylabel('Model', fontsize=10)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlim([0, 1])
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
