from typing import Dict, List, Tuple
import numpy as np
from config.settings import ValidatorConfig
from src.utils import get_top_k_features

class XAIValidator:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.tolerance = config.sum_tolerance
    
    def verify_sum_conservation(
        self,
        contributions: Dict[str, float],
        base_value: float,
        prediction: float,
        tolerance: float = None
    ) -> Tuple[bool, float]:
        """
        Verify SHAP sum conservation property.
        
        For tree-based SHAP: sum(contributions) + base_value ≈ prediction
        
        Args:
            contributions: Feature contributions
            base_value: Expected model output
            prediction: Actual model prediction
            tolerance: Tolerance for equality check
            
        Returns:
            (is_valid, computed_sum)
        """
        tolerance = tolerance or self.tolerance
        computed_sum = sum(contributions.values()) + base_value
        
        is_valid = abs(computed_sum - prediction) < tolerance
        return is_valid, computed_sum
    
    def verify_stability(
        self,
        explanations_t1: Dict[str, float],
        explanations_t2: Dict[str, float],
        k: int = 5
    ) -> Tuple[float, List[str]]:
        """
        Verify explanation stability across runs.
        
        Uses Jaccard similarity on top-k features.
        
        Args:
            explanations_t1: Features from first explanation
            explanations_t2: Features from second explanation
            k: Number of top features to compare
            
        Returns:
            (jaccard_similarity, top_features)
        """
        top_features_t1 = set([f[0] for f in get_top_k_features(explanations_t1, k)])
        top_features_t2 = set([f[0] for f in get_top_k_features(explanations_t2, k)])
        
        jaccard = len(top_features_t1 & top_features_t2) / len(top_features_t1 | top_features_t2)
        
        common_features = list(top_features_t1 & top_features_t2)
        return jaccard, common_features
    
    def compute_clarity_score(self, generated_text: str) -> float:
        """
        Compute clarity score for generated explanation.
        
        Simple heuristic: based on sentence length, vocabulary simplicity, etc.
        
        Args:
            generated_text: Natural language explanation
            
        Returns:
            Clarity score (0-100)
        """
        sentences = generated_text.split(".")
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s])
        
        # Penalize very long sentences
        clarity = 100 - (avg_sentence_length - 10) * 2
        clarity = max(0, min(100, clarity))  # Clamp to 0-100
        
        return clarity
    
    def validate_all(
        self,
        explanation: Dict[str, float],
        generated_text: str,
        base_value: float,
        prediction: float,
        method: str = "shap",
    ) -> Dict:
        """
        Run all validation checks.

        Returns:
            {
                "sum_conservation_valid": bool,   # only for SHAP
                "computed_sum": float,            # only for SHAP
                "clarity_score": float,
                ...
            }
        """
        results: Dict[str, Any] = {}

        # only SHAP should be checked for sum conservation
        if self.config.verify_sum_conservation and method == "shap":
            is_valid, computed = self.verify_sum_conservation(
                explanation, base_value, prediction
            )
            results["sum_conservation_valid"] = bool(is_valid)
            results["computed_sum"] = float(computed)

        # always compute clarity score, and make sure it's a plain float
        results["clarity_score"] = float(self.compute_clarity_score(generated_text))

        return results

