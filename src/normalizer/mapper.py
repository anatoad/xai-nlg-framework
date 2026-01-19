"""
Feature Mapper for converting XAI contributions to natural language statements.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FeatureStatement:
    """Represents a feature mapped to a natural language statement."""
    feature: str
    value: float
    normalized_value: float
    rank: int
    direction: str  # "positive", "negative", "neutral"
    magnitude: str  # "high", "medium", "low"
    statement: str


class FeatureMapper:
    """
    Maps feature contributions to natural language statements.
    
    Uses templates to generate human-readable descriptions of
    how each feature contributes to the prediction.
    """
    
    def __init__(self, custom_templates: Dict[str, str] = None):
        """
        Initialize the mapper.
        
        Args:
            custom_templates: Optional custom templates for specific features
        """
        self.custom_templates = custom_templates or {}
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default templates for different directions and magnitudes."""
        self.templates = {
            "positive": {
                "high": "The {feature} strongly supports the prediction (contribution: {value:.4f})",
                "medium": "The {feature} moderately supports the prediction (contribution: {value:.4f})",
                "low": "The {feature} slightly supports the prediction (contribution: {value:.4f})"
            },
            "negative": {
                "high": "The {feature} strongly opposes the prediction (contribution: {value:.4f})",
                "medium": "The {feature} moderately opposes the prediction (contribution: {value:.4f})",
                "low": "The {feature} slightly opposes the prediction (contribution: {value:.4f})"
            },
            "neutral": {
                "high": "The {feature} has minimal impact on the prediction",
                "medium": "The {feature} has minimal impact on the prediction",
                "low": "The {feature} has minimal impact on the prediction"
            }
        }
    
    def get_direction(self, value: float, threshold: float = 1e-6) -> str:
        """Determine the direction of contribution."""
        if abs(value) < threshold:
            return "neutral"
        return "positive" if value > 0 else "negative"
    
    def get_magnitude(self, normalized_value: float) -> str:
        """Determine the magnitude category."""
        if normalized_value >= 0.66:
            return "high"
        elif normalized_value >= 0.33:
            return "medium"
        else:
            return "low"
    
    def map_single(
        self,
        feature: str,
        value: float,
        normalized_value: float,
        rank: int
    ) -> FeatureStatement:
        """
        Map a single feature to a statement.
        
        Args:
            feature: Feature name
            value: Raw contribution value
            normalized_value: Normalized value [0, 1]
            rank: Rank among all features (1 = most important)
            
        Returns:
            FeatureStatement object
        """
        direction = self.get_direction(value)
        magnitude = self.get_magnitude(normalized_value)
        
        # Check for custom template
        if feature in self.custom_templates:
            template = self.custom_templates[feature]
        else:
            template = self.templates[direction][magnitude]
        
        # Format feature name for display (replace underscores with spaces)
        display_name = feature.replace("_", " ")
        
        statement = template.format(
            feature=display_name,
            value=value,
            normalized=normalized_value
        )
        
        return FeatureStatement(
            feature=feature,
            value=value,
            normalized_value=normalized_value,
            rank=rank,
            direction=direction,
            magnitude=magnitude,
            statement=statement
        )
    
    def map_features(
        self,
        ranked_features: List[Tuple[str, float]],
        normalized_values: Dict[str, float]
    ) -> List[FeatureStatement]:
        """
        Map multiple ranked features to statements.
        
        Args:
            ranked_features: List of (feature, value) sorted by importance
            normalized_values: Dictionary of normalized values
            
        Returns:
            List of FeatureStatement objects
        """
        statements = []
        
        for rank, (feature, value) in enumerate(ranked_features, start=1):
            norm_value = normalized_values.get(feature, 0.5)
            statement = self.map_single(feature, value, norm_value, rank)
            statements.append(statement)
        
        return statements
    
    def get_summary_context(
        self,
        statements: List[FeatureStatement],
        prediction: str,
        method: str
    ) -> Dict:
        """
        Create a context dictionary for NLG generation.
        
        Args:
            statements: List of FeatureStatement objects
            prediction: Model prediction (as string)
            method: XAI method used ("shap" or "lime")
            
        Returns:
            Context dictionary for NLG generators
        """
        supporting = [s for s in statements if s.direction == "positive"]
        opposing = [s for s in statements if s.direction == "negative"]
        
        return {
            "prediction": prediction,
            "method": method.upper(),
            "features": [s.feature for s in statements],
            "values": [s.value for s in statements],
            "directions": [
                "supports" if s.direction == "positive" 
                else "contradicts" if s.direction == "negative"
                else "neutral"
                for s in statements
            ],
            "statements": [s.statement for s in statements],
            "n_supporting": len(supporting),
            "n_opposing": len(opposing),
            "top_supporting": supporting[0].feature if supporting else None,
            "top_opposing": opposing[0].feature if opposing else None
        }