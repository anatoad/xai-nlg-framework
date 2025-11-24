from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class FeatureStatement:
    """ Represents a feature-to-statement mapping """
    feature: str
    value: float
    rank: int
    direction: str  # "positive", "negative", "neutral"
    statement: str

class FeatureMapper:
    """Maps features to natural language statements."""
    
    def __init__(self, feature_templates: Dict[str, Dict] = None):
        """
        Initialize mapper with feature templates.
        
        Args:
            feature_templates: Dictionary mapping feature names to template info
        """
        self.feature_templates = feature_templates or {}
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default templates for common scenarios."""
        self.templates = {
            "positive": {
                "high": "The {feature} (value: {value:.2f}) strongly supports the prediction.",
                "medium": "The {feature} (value: {value:.2f}) moderately supports the prediction.",
                "low": "The {feature} (value: {value:.2f}) slightly supports the prediction.",
            },
            "negative": {
                "high": "The {feature} (value: {value:.2f}) strongly contradicts the prediction.",
                "medium": "The {feature} (value: {value:.2f}) moderately contradicts the prediction.",
                "low": "The {feature} (value: {value:.2f}) slightly contradicts the prediction.",
            }
        }
    
    def determine_direction(self, value: float, threshold: float = 0.01) -> str:
        """Determine direction of feature contribution."""
        if abs(value) < threshold:
            return "neutral"
        return "positive" if value > 0 else "negative"
    
    def determine_magnitude(self, norm_value: float) -> str:
        """Classify magnitude as high, medium, or low."""
        if norm_value > 0.66:
            return "high"
        if norm_value > 0.33:
            return "medium"
        return "low"
    
    def map_feature_to_statement(
        self,
        feature: str,
        value: float,
        normalized_value: float,
        rank: int
    ) -> FeatureStatement:
        """
        Map a feature to a natural language statement.
        
        Args:
            feature: Feature name
            value: Original contribution value
            normalized_value: Normalized value (0-1)
            rank: Rank of feature (1st, 2nd, etc.)
            
        Returns:
            FeatureStatement object
        """
        direction = self.determine_direction(value)
        magnitude = self.determine_magnitude(abs(normalized_value))
        
        if direction == "neutral":
            statement = f"The {feature} has a neutral effect."
        else:
            template_key = "positive" if value > 0 else "negative"
            statement = self.templates[template_key][magnitude].format(
                feature=feature,
                value=value
            )
        
        return FeatureStatement(
            feature=feature,
            value=value,
            rank=rank,
            direction=direction,
            statement=statement
        )
    
    def map_features(
        self,
        ranked_features: List[Tuple[str, float]],
        normalized_contributions: Dict[str, float]
    ) -> List[FeatureStatement]:
        """
        Map multiple ranked features to statements.
        
        Args:
            ranked_features: List of (feature, value) sorted by importance
            normalized_contributions: Dictionary of normalized values
            
        Returns:
            List of FeatureStatement objects
        """
        statements = []
        for rank, (feature, value) in enumerate(ranked_features, 1):
            norm_value = normalized_contributions.get(feature, 0.0)
            statement = self.map_feature_to_statement(feature, value, norm_value, rank)
            statements.append(statement)
        return statements
