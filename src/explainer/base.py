"""
Base class for XAI explainers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import pandas as pd


class BaseExplainer(ABC):
    """Abstract base class for explainers (SHAP, LIME, etc.)."""
    
    def __init__(self, model, data: np.ndarray, feature_names: List[str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ML model with predict/predict_proba methods
            data: Training data for background (numpy array)
            feature_names: List of feature names
        """
        self.model = model
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        
    @abstractmethod
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Generate explanation for a single instance.
        
        Args:
            instance: 1D array of feature values
            
        Returns:
            Dictionary mapping feature names to contribution values
        """
        pass
    
    @abstractmethod
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        """
        Generate explanations for multiple instances.
        
        Args:
            instances: 2D array of shape (n_samples, n_features)
            
        Returns:
            List of explanation dictionaries
        """
        pass
    
    def get_base_value(self) -> float:
        """Get the base/expected value for the explainer."""
        return 0.0
    
    def to_dataframe(
        self,
        explanation: Dict[str, float],
        instance: np.ndarray = None,
        sort_by: str = "absolute"
    ) -> pd.DataFrame:
        """
        Convert explanation to a pandas DataFrame.
        
        Args:
            explanation: Feature contribution dictionary
            instance: Optional instance values
            sort_by: "absolute" (by |value|), "value", or "name"
            
        Returns:
            DataFrame with columns: feature, contribution, [value]
        """
        data = {
            'feature': list(explanation.keys()),
            'contribution': list(explanation.values())
        }
        
        if instance is not None:
            data['value'] = [instance[self.feature_names.index(f)] 
                           for f in explanation.keys()]
        
        df = pd.DataFrame(data)
        
        if sort_by == "absolute":
            df = df.reindex(df['contribution'].abs().sort_values(ascending=False).index)
        elif sort_by == "value":
            df = df.sort_values('contribution', ascending=False)
        elif sort_by == "name":
            df = df.sort_values('feature')
            
        return df.reset_index(drop=True)