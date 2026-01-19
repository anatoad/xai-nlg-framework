"""
Validator for XAI explanations and generated text.
"""
import numpy as np
from typing import Dict, List, Tuple, Any


class ExplanationValidator:
    """
    Validates XAI explanations and generated text.
    
    Provides:
    - SHAP sum conservation check
    - Clarity/readability scoring
    - Feature coverage analysis
    - Stability metrics
    """
    
    def __init__(self, config=None):
        """
        Initialize validator.
        
        Args:
            config: ValidatorConfig
        """
        self.config = config
        self.sum_tolerance = config.sum_tolerance if config else 0.1
        self.min_clarity = config.min_clarity_score if config else 40.0
    
    def verify_shap_sum(
        self,
        contributions: Dict[str, float],
        base_value: float,
        prediction: float
    ) -> Tuple[bool, float, float]:
        """
        Verify SHAP sum conservation property.
        
        For SHAP: sum(contributions) + base_value ≈ prediction
        
        Args:
            contributions: Feature contribution dictionary
            base_value: SHAP expected value
            prediction: Model prediction (probability or score)
            
        Returns:
            (is_valid, computed_sum, difference)
        """
        contrib_sum = sum(contributions.values())
        computed = contrib_sum + base_value
        difference = abs(computed - prediction)
        
        is_valid = difference <= self.sum_tolerance
        
        return is_valid, computed, difference
    
    def compute_clarity_score(self, text: str) -> float:
        """
        Compute readability/clarity score for generated text.
        
        Uses a simplified Flesch-like metric based on:
        - Average sentence length (penalize very long sentences)
        - Word complexity (average word length)
        
        Args:
            text: Generated explanation text
            
        Returns:
            Clarity score (0-100)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Split into sentences
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if not sentences:
            return 50.0
        
        # Calculate metrics
        words = text.split()
        if not words:
            return 50.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(w) for w in words])
        
        # Score calculation (higher is better)
        # Optimal sentence length: 15-20 words
        # Optimal word length: 4-6 characters
        
        sentence_penalty = abs(avg_sentence_length - 17.5) * 2
        word_penalty = abs(avg_word_length - 5) * 5
        
        score = 100 - sentence_penalty - word_penalty
        
        # Clamp to 0-100
        return max(0.0, min(100.0, score))
    
    def check_feature_coverage(
        self,
        text: str,
        features: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Check how many top features are mentioned in the text.
        
        Args:
            text: Generated explanation text
            features: List of feature names (in ranked order)
            top_k: Number of top features to check
            
        Returns:
            Dictionary with coverage metrics
        """
        text_lower = text.lower()
        top_features = features[:top_k]
        
        mentioned = []
        missing = []
        
        for feature in top_features:
            # Check various forms of the feature name
            feature_lower = feature.lower()
            feature_clean = feature_lower.replace("_", " ")
            
            if feature_lower in text_lower or feature_clean in text_lower:
                mentioned.append(feature)
            else:
                missing.append(feature)
        
        coverage = len(mentioned) / len(top_features) if top_features else 0.0
        
        return {
            "coverage_score": coverage,
            "mentioned": mentioned,
            "missing": missing,
            "top_k": top_k
        }
    
    def compute_stability(
        self,
        explanations: List[Dict[str, float]],
        k: int = 5
    ) -> float:
        """
        Compute stability (Jaccard similarity) of top-k features across explanations.
        
        Args:
            explanations: List of explanation dictionaries
            k: Number of top features to compare
            
        Returns:
            Average Jaccard similarity (0-1)
        """
        if len(explanations) < 2:
            return 1.0
        
        # Get top-k feature sets
        top_k_sets = []
        for exp in explanations:
            ranked = sorted(exp.items(), key=lambda x: abs(x[1]), reverse=True)
            top_k_sets.append(set(f for f, _ in ranked[:k]))
        
        # Compute pairwise Jaccard
        similarities = []
        for i in range(len(top_k_sets)):
            for j in range(i + 1, len(top_k_sets)):
                intersection = len(top_k_sets[i] & top_k_sets[j])
                union = len(top_k_sets[i] | top_k_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 1.0
    
    def validate_explanation(
        self,
        contributions: Dict[str, float],
        generated_text: str,
        base_value: float = 0.0,
        prediction: float = 0.0,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Run all validation checks on an explanation.
        
        Args:
            contributions: Feature contributions dictionary
            generated_text: Generated NLG text
            base_value: SHAP base value (or 0 for LIME)
            prediction: Model prediction
            method: "shap" or "lime"
            
        Returns:
            Dictionary with all validation results
        """
        results = {}
        
        # SHAP sum conservation (only for SHAP)
        if method.lower() == "shap":
            is_valid, computed, diff = self.verify_shap_sum(
                contributions, base_value, prediction
            )
            results["sum_conservation"] = {
                "valid": is_valid,
                "computed_sum": computed,
                "expected": prediction,
                "difference": diff
            }
        
        # Clarity score
        clarity = self.compute_clarity_score(generated_text)
        results["clarity"] = {
            "score": clarity,
            "passes_threshold": clarity >= self.min_clarity
        }
        
        # Feature coverage
        features = list(contributions.keys())
        ranked_features = sorted(features, key=lambda f: abs(contributions[f]), reverse=True)
        coverage = self.check_feature_coverage(generated_text, ranked_features)
        results["coverage"] = coverage
        
        # Overall pass/fail
        results["valid"] = (
            (method.lower() != "shap" or results.get("sum_conservation", {}).get("valid", True)) and
            results["clarity"]["passes_threshold"] and
            results["coverage"]["coverage_score"] >= 0.4  # At least 40% of top features mentioned
        )
        
        return results