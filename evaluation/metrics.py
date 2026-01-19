import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr
from collections import defaultdict
import time

@dataclass
class MetricsResult:
    instance_id: str
    fidelity_score: float
    sign_match_score: float
    clarity_score: float
    coverage_topk: float
    shap_conservation_valid: bool
    shap_conservation_error: float
    jaccard_score: Optional[float] = None
    spearman_correlation: Optional[float] = None
    ranking_stability: Optional[float] = None
    dataset: str = ""
    method: str = ""  # SHAP/LIME
    
    def to_dict(self) -> Dict:
        return {
            'instance_id': self.instance_id,
            'fidelity_score': self.fidelity_score,
            'sign_match_score': self.sign_match_score,
            'clarity_score': self.clarity_score,
            'coverage_topk': self.coverage_topk,
            'shap_conservation_valid': self.shap_conservation_valid,
            'shap_conservation_error': self.shap_conservation_error,
            'jaccard_score': self.jaccard_score,
            'spearman_correlation': self.spearman_correlation,
            'ranking_stability': self.ranking_stability,
            'dataset': self.dataset,
            'method': self.method,
        }

class FidelityMetrics:
    # fidelity metrics for the explanations
    
    @staticmethod
    def sign_match_score(
        true_contributions: Dict[str, float],
        text_features: List[str],
        text_content: str,
        top_k: int = 5
    ) -> Tuple[float, Dict]:
        """
        Calculates the percentage of the correct signs (positive/negative)
        in the generated textual explanation.
        
        Args:
            true_contributions: Dict {feature: value} from XAI
            text_features: list of features mentioned in the text
            text_content: generated textual explanation
            top_k: number of top k features to be verified
            
        Returns:
            (score: float, details: Dict)
        """
        # sort contributions by absolute values
        sorted_contribs = sorted(
            true_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_k_true = sorted_contribs[:top_k]
        
        correct_signs = 0
        correct_order = 0
        
        # keywords for the signs (positive/ negative)
        positive_keywords = ['support', 'strong', 'increase', 'positive', 'high']
        negative_keywords = ['contradict', 'negative', 'low', 'decrease', 'reduce']
        
        for idx, (feature, value) in enumerate(top_k_true):
            # check if feature appears in the text
            if feature.lower() in text_content.lower():
                # extract context
                pattern = r'.{0,50}' + re.escape(feature) + r'.{0,50}'
                matches = re.findall(pattern, text_content.lower(), re.IGNORECASE)
                
                if matches:
                    context = matches[0]
                    
                    # check sign
                    if value > 0:
                        has_correct_sign = any(kw in context for kw in positive_keywords)
                    else:
                        has_correct_sign = any(kw in context for kw in negative_keywords)
                    
                    if has_correct_sign:
                        correct_signs += 1
        
        # check order
        feature_order_in_text = [f for f in text_features if f in [feat for feat, _ in top_k_true]]
        true_order = [feat for feat, _ in top_k_true]
        
        if len(feature_order_in_text) > 1:
            matches_order = 0
            for i in range(len(feature_order_in_text) - 1):
                true_idx = true_order.index(feature_order_in_text[i])
                if i + 1 < len(feature_order_in_text):
                    next_true_idx = true_order.index(feature_order_in_text[i + 1])
                    if true_idx < next_true_idx:
                        matches_order += 1
            correct_order = matches_order / max(1, len(feature_order_in_text) - 1)
        
        # Final score: mean of correct_signs and correct_order
        score = (correct_signs / max(1, len(top_k_true)) + correct_order) / 2
        
        details = {
            'correct_signs': correct_signs,
            'total_features': len(top_k_true),
            'order_matches': correct_order,
            'mentioned_features': len(feature_order_in_text)
        }
        
        return min(1.0, score), details
    
    @staticmethod
    def shap_sum_conservation(
        shap_values: Dict[str, float],
        prediction: float,
        base_value: float,
        tolerance: float = 1e-5
    ) -> Tuple[bool, float]:
        """
        Check SHAP sum conservation
        
        Args:
            shap_values: Dict {feature: shap_value}
            prediction: model prediction F(x)
            base_value: base value E[F]
            tolerance: tolerance
            
        Returns:
            (is_valid: bool, error: float)
        """
        shap_sum = sum(shap_values.values())
        expected_sum = prediction - base_value
        error = abs(shap_sum - expected_sum)
        is_valid = error <= tolerance
        
        return is_valid, error
    
    @staticmethod
    def magnitude_alignment(
        normalized_contributions: Dict[str, float],
        text_magnitudes: Dict[str, str]  # {feature: 'high'/'medium'/'low'}
    ) -> float:
        """
        Check if magnitudes from the text align with the normalized values.
        
        Args:
            normalized_contributions: Dict {feature: normalized_value in [0,1]}
            text_magnitudes: Dict {feature: magnitude string}
            
        Returns:
            alignment_score: float in [0, 1]
        """
        thresholds = {'high': 0.66, 'medium': 0.33, 'low': 0.0}
        magnitude_ranges = {
            'high': (0.66, 1.0),
            'medium': (0.33, 0.66),
            'low': (0.0, 0.33)
        }
        
        matches = 0
        total = 0
        
        for feature, text_mag in text_magnitudes.items():
            if feature in normalized_contributions:
                total += 1
                value = abs(normalized_contributions[feature])
                min_val, max_val = magnitude_ranges.get(text_mag, (0, 1))
                if min_val <= value <= max_val:
                    matches += 1
        
        return matches / max(1, total) if total > 0 else 0.0

class RobustnessMetrics:
    """ Robustness metrics """
    
    @staticmethod
    def jaccard_at_k(
        list_a: List[str],
        list_b: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate the Jaccard similarity @k between two lists.
        J@k = |A_k n B_k| / |A_k u B_k|
        """
        set_a = set(list_a[:k])
        set_b = set(list_b[:k])
        
        if len(set_a.union(set_b)) == 0:
            return 1.0
        
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    
    @staticmethod
    def ranking_consistency(
        ranks_a: List[Tuple[str, float]],
        ranks_b: List[Tuple[str, float]]
    ) -> float:
        """
        Calculate Spearman Rank Correlation between two rankings.
        """
        # normalize rankings
        features_a = [f for f, _ in ranks_a]
        features_b = [f for f, _ in ranks_b]
        common_features = set(features_a).intersection(set(features_b))
        
        if len(common_features) < 2:
            return 1.0
        
        ranks_a_dict = {f: i for i, (f, _) in enumerate(ranks_a)}
        ranks_b_dict = {f: i for i, (f, _) in enumerate(ranks_b)}
        
        a_ranks = [ranks_a_dict[f] for f in common_features]
        b_ranks = [ranks_b_dict[f] for f in common_features]
        
        corr, _ = spearmanr(a_ranks, b_ranks)
        return corr if not np.isnan(corr) else 1.0
    
    @staticmethod
    def contribution_stability(
        contributions_runs: List[Dict[str, float]],
        top_k: int = 5
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate stability of contributions based on more runs.
        
        Args:
            contributions_runs: List[Dict] with contributions from multiple runs
            top_k: features to be analyzed
            
        Returns:
            (stability_score, instability_flags)
        """
        if len(contributions_runs) < 2:
            return 1.0, {}
        
        # collecta all top k features
        all_features = set()
        for contrib in contributions_runs:
            sorted_contrib = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
            all_features.update([f for f, _ in sorted_contrib[:top_k]])
        
        instability_flags = {}
        total_variance = 0.0
        
        for feature in all_features:
            values = []
            for contrib in contributions_runs:
                values.append(contrib.get(feature, 0.0))
            
            std_dev = np.std(values)
            mean_val = np.mean(values)
            
            # coefficient of variation
            cv = (std_dev / abs(mean_val)) if mean_val != 0 else 0
            
            if cv > 0.1:  # 10% instability
                instability_flags[feature] = cv
            
            total_variance += cv
        
        stability_score = 1.0 - (total_variance / max(1, len(all_features)))
        
        return max(0.0, stability_score), instability_flags

class UsefulnessMetrics:
    """ Usefulness metrics """
    
    @staticmethod
    def clarity_score(
        text: str,
        features_mentioned: List[str],
        top_k: int = 5
    ) -> float:
        """
        Calculate clarity score for text.
        Factors:
        - length (3-100 words = ideal)
        - repetitions (lowers score)
        - presence of key XAI words
        - lexical simplicity
        """
        # normalize text
        text_lower = text.lower()
        words = text_lower.split()
        
        # Factor 1: optimal length
        word_count = len(words)
        if word_count < 3:
            length_score = 0.2
        elif word_count < 50:
            length_score = 1.0
        elif word_count < 100:
            length_score = 0.9
        else:
            length_score = 0.5
        
        # Factor 2: repetitions
        unique_words = len(set(words))
        repetition_score = unique_words / max(1, word_count)
        
        # Factor 3: presence of keywords
        xai_keywords = [
            'support', 'contradict', 'influence', 'impact', 'important',
            'strong', 'moderate', 'weak', 'positive', 'negative',
            'high', 'low', 'feature', 'predict', 'contribute'
        ]
        keyword_mentions = sum(1 for kw in xai_keywords if kw in text_lower)
        keyword_score = min(1.0, keyword_mentions / 3)  # minimum of 3 keywords
        
        # Factor 4: lexical simplicity (avg word length)
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        if avg_word_length < 8:
            lexical_score = 1.0
        elif avg_word_length < 12:
            lexical_score = 0.8
        else:
            lexical_score = 0.5
        
        # Final score
        clarity = (
            0.25 * length_score +
            0.25 * repetition_score +
            0.25 * keyword_score +
            0.25 * lexical_score
        ) * 100
        
        return clarity
    
    @staticmethod
    def coverage_topk(
        text: str,
        features_topk: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate the percentage of top-k features mentioned in the text.
        """
        text_lower = text.lower()
        mentioned = sum(
            1 for feature in features_topk[:k]
            if feature.lower() in text_lower
        )
        
        return (mentioned / k) * 100 if k > 0 else 0.0


if __name__ == '__main__':
    # Test basic metrics
    test_contrib = {'feature_a': 0.5, 'feature_b': -0.3, 'feature_c': 0.1}
    print(f"Sum conservation test: {FidelityMetrics.shap_sum_conservation(test_contrib, 0.7, 0.4)}")
    
    test_text = "The feature_a strongly supports the prediction, while feature_b contradicts it."
    score, details = FidelityMetrics.sign_match_score(
        test_contrib,
        ['feature_a', 'feature_b'],
        test_text
    )
    print(f"Sign match score: {score}, details: {details}")
    
    clarity = UsefulnessMetrics.clarity_score(test_text, ['feature_a', 'feature_b'])
    print(f"Clarity score: {clarity:.2f}")
