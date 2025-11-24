from typing import Dict, List
import numpy as np
from config.settings import NormalizerConfig
from src.utils import normalize_values

class Normalizer:
    """
    Normalizes feature contributions from explainers
    """
    
    def __init__(self, config: NormalizerConfig):
        self.config = config
        self.scale_method = config.scale_method
    
    def normalize_contributions(self, contributions: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize feature contributions using configured method.
        
        Args:
            contributions: Dictionary of feature names to values
            
        Returns:
            Normalized contributions
        """
        return normalize_values(contributions, method=self.scale_method)
    
    def group_small_features(
        self,
        contributions: Dict[str, float],
        threshold: float = None
    ) -> Dict[str, float]:
        """
        Group features with contributions below threshold into 'Others'.
        
        Args:
            contributions: Dictionary of features to values
            threshold: Grouping threshold (uses config if None)
            
        Returns:
            Contributions with small features grouped
        """
        threshold = threshold or self.config.feature_grouping_threshold
        grouped = {}
        others_sum = 0.0
        
        for feature, value in contributions.items():
            if abs(value) < threshold:
                others_sum += value
            else:
                grouped[feature] = value
        
        if others_sum != 0:
            grouped["others"] = others_sum
        
        return grouped
    
    def rank_features(self, contributions: Dict[str, float]) -> List[tuple]:
        """
        Rank features by absolute contribution value.
        
        Args:
            contributions: Dictionary of features to values
            
        Returns:
            Sorted list of (feature, value) tuples
        """
        return sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
