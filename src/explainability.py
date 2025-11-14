"""
SHAP Explainability Module for Model Interpretability
Provides parallel execution for SHAP value computation
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import config


class SHAPExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, background_data, feature_names=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model with predict method
            background_data: Background dataset for SHAP (numpy array)
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit(self):
        """Initialize the SHAP explainer"""
        # Sample background data if too large
        if len(self.background_data) > config.SHAP_SAMPLE_SIZE:
            indices = np.random.choice(
                len(self.background_data), 
                config.SHAP_SAMPLE_SIZE, 
                replace=False
            )
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data
        
        # Use KernelExplainer for model-agnostic explanations
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba, 
            background_sample
        )
        
        return self
    
    def explain(self, X, parallel=True):
        """
        Compute SHAP values for input data
        
        Args:
            X: Input data to explain (numpy array)
            parallel: Whether to use parallel computation
        
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.fit()
        
        if parallel and len(X) > 1:
            self.shap_values = self._explain_parallel(X)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def _explain_parallel(self, X, n_workers=4):
        """
        Compute SHAP values in parallel
        
        Args:
            X: Input data
            n_workers: Number of parallel workers
        
        Returns:
            Combined SHAP values
        """
        # Split data into chunks
        chunk_size = max(1, len(X) // n_workers)
        chunks = [X[i:i+chunk_size] for i in range(0, len(X), chunk_size)]
        
        shap_results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit jobs
            futures = {
                executor.submit(self.explainer.shap_values, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    shap_results.append((chunk_idx, result))
                except Exception as e:
                    print(f"Error computing SHAP for chunk {chunk_idx}: {e}")
        
        # Sort by chunk index and combine
        shap_results.sort(key=lambda x: x[0])
        
        # Handle multi-class case
        if isinstance(shap_results[0][1], list):
            combined = []
            for class_idx in range(len(shap_results[0][1])):
                class_values = np.vstack([r[1][class_idx] for r in shap_results])
                combined.append(class_values)
        else:
            combined = np.vstack([r[1] for r in shap_results])
        
        return combined
    
    def plot_summary(self, X=None, class_idx=1, max_display=None):
        """
        Generate SHAP summary plot
        
        Args:
            X: Input data (if None, uses data from last explain call)
            class_idx: Class index for multi-class (default: 1 for churn class)
            max_display: Maximum features to display
        """
        if self.shap_values is None and X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain() first.")
        
        max_display = max_display or config.SHAP_MAX_DISPLAY
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            values_to_plot = self.shap_values[class_idx]
        else:
            values_to_plot = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            values_to_plot,
            features=X if X is not None else self.background_data[:len(values_to_plot)],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waterfall(self, instance_idx, X, class_idx=1):
        """
        Generate SHAP waterfall plot for a single instance
        
        Args:
            instance_idx: Index of instance to explain
            X: Input data
            class_idx: Class index for multi-class
        """
        if self.shap_values is None:
            self.explain(X)
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            values = self.shap_values[class_idx][instance_idx]
        else:
            values = self.shap_values[instance_idx]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=values,
            base_values=self.explainer.expected_value[class_idx] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            data=X[instance_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        return plt.gcf()
    
    def get_feature_importance(self, class_idx=1):
        """
        Get feature importance based on mean absolute SHAP values
        
        Args:
            class_idx: Class index for multi-class
        
        Returns:
            Dict with feature importance
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain() first.")
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            values = self.shap_values[class_idx]
        else:
            values = self.shap_values
        
        # Compute mean absolute SHAP values
        importance = np.abs(values).mean(axis=0)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}


def explain_prediction_parallel(model, X, background_data, feature_names=None):
    """
    Convenience function for parallel SHAP explanation
    
    Args:
        model: Trained model
        X: Input data to explain
        background_data: Background dataset
        feature_names: Feature names
    
    Returns:
        SHAP values
    """
    explainer = SHAPExplainer(model, background_data, feature_names)
    return explainer.explain(X, parallel=True)
