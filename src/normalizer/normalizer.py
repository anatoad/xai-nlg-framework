"""
Normalizer for XAI contributions.

Normalizes and ranks feature contributions from SHAP/LIME.
"""
import numpy as np
from typing import Dict, List, Tuple


class Normalizer:
    """
    Normalizes and processes feature contributions.
    
    Provides:
    - Min-max normalization to [0, 1]
    - Feature ranking by absolute contribution
    - Grouping of small contributions
    """
    
    def __init__(self, config=None):
        """
        Initialize normalizer.
        
        Args:
            config: NormalizerConfig (optional)
        """
        self.config = config
        self.scale_method = config.scale_method if config else "minmax"
        self.grouping_threshold = config.feature_grouping_threshold if config else 0.05
    
    def normalize(self, contributions: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize contributions to [0, 1] range based on absolute values.
        
        Args:
            contributions: Dictionary of feature->value
            
        Returns:
            Normalized contributions
        """
        if not contributions:
            return {}
        
        values = np.array(list(contributions.values()))
        abs_values = np.abs(values)
        
        if self.scale_method == "minmax":
            # Min-max normalization on absolute values
            min_val = abs_values.min()
            max_val = abs_values.max()
            
            if max_val - min_val > 1e-10:
                normalized = (abs_values - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(abs_values) * 0.5
        
        elif self.scale_method == "standard":
            # Z-score normalization, then sigmoid to [0, 1]
            mean = abs_values.mean()
            std = abs_values.std()
            if std > 1e-10:
                z_scores = (abs_values - mean) / std
                normalized = 1 / (1 + np.exp(-z_scores))  # Sigmoid
            else:
                normalized = np.ones_like(abs_values) * 0.5
        
        else:  # robust
            # Median and IQR based
            median = np.median(abs_values)
            q75, q25 = np.percentile(abs_values, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-10:
                normalized = (abs_values - q25) / iqr
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = np.ones_like(abs_values) * 0.5
        
        return {
            name: float(normalized[i])
            for i, name in enumerate(contributions.keys())
        }
    
    def rank_features(
        self,
        contributions: Dict[str, float],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Rank features by absolute contribution value.
        
        Args:
            contributions: Dictionary of feature->value
            top_k: Return only top k features (None for all)
            
        Returns:
            List of (feature_name, contribution) sorted by |contribution|
        """
        ranked = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def group_small_features(
        self,
        contributions: Dict[str, float],
        threshold: float = None
    ) -> Dict[str, float]:
        """
        Group features with small contributions into "others".
        
        Args:
            contributions: Dictionary of feature->value
            threshold: Minimum absolute value to keep separate
            
        Returns:
            Contributions with small features grouped
        """
        threshold = threshold or self.grouping_threshold
        
        grouped = {}
        others_sum = 0.0
        
        # Normalize to find threshold relative to max
        normalized = self.normalize(contributions)
        
        for feature, value in contributions.items():
            if normalized.get(feature, 0) < threshold:
                others_sum += value
            else:
                grouped[feature] = value
        
        if abs(others_sum) > 1e-10:
            grouped["_others"] = others_sum
        
        return grouped
    
    def get_direction(self, value: float) -> str:
        """
        Get the direction of a contribution.
        
        Args:
            value: Contribution value
            
        Returns:
            "positive", "negative", or "neutral"
        """
        if abs(value) < 1e-6:
            return "neutral"
        return "positive" if value > 0 else "negative"
    
    def get_magnitude(self, normalized_value: float) -> str:
        """
        Get the magnitude category of a normalized value.
        
        Args:
            normalized_value: Value in [0, 1]
            
        Returns:
            "high", "medium", or "low"
        """
        if normalized_value >= 0.66:
            return "high"
        elif normalized_value >= 0.33:
            return "medium"
        else:
            return "low"