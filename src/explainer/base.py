from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseExplainer(ABC):
    def __init__(self, model, data, feature_names: list):
        """
        Initialize the explainer with model
        Args:
            model: trained model with predict method
            data: training data (numpy array or DataFrame)
            feature_names: list of feature names
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    @abstractmethod
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Generate explanation for a single instance        
        Returns:
            Dictionary mapping feature names to contribution values
        """
        pass
    
    @abstractmethod
    def explain_batch(self, instances: np.ndarray) -> list:
        """ Generate explanations for a batch """
        pass
    
    def get_explanation_metadata(self, explanation: Dict[str, float]) -> Dict[str, Any]:
        """ extract metadata about explanation """
        values = list(explanation.values())
        return {
            "sum": sum(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
            "positive_count": sum(1 for v in values if v > 0),
            "negative_count": sum(1 for v in values if v < 0),
        }
