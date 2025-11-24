import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_top_k_features(
    feature_values: Dict[str, float],
    k: int = 5,
    include_negative: bool = True
) -> List[Tuple[str, float]]:
    """
    Extract top-k features by absolute value.
    
    Args:
        feature_values: Dictionary of feature names to contribution values
        k: Number of top features to return
        include_negative: If True, considers absolute values
        
    Returns:
        List of (feature_name, value) tuples sorted by importance
    """
    if include_negative:
        sorted_features = sorted(
            feature_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    else:
        sorted_features = sorted(
            feature_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
    return sorted_features[:k]

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def normalize_values(
    values: Dict[str, float],
    method: str = "minmax"
) -> Dict[str, float]:
    """
    Normalize feature contribution values.
    
    Args:
        values: dictionary of feature names to values
        method: "minmax", "standard", or "robust"
        
    Returns:
        Normalized values dictionary
    """
    vals = np.array(list(values.values()))
    
    if method == "minmax":
        min_val, max_val = vals.min(), vals.max()
        if max_val == min_val:
            normalized = {k: 0.5 for k in values.keys()}
        else:
            normalized = {
                k: (v - min_val) / (max_val - min_val)
                for k, v in values.items()
            }
    elif method == "standard":
        mean, std = vals.mean(), vals.std()
        if std == 0:
            normalized = {k: 0.0 for k in values.keys()}
        else:
            normalized = {k: (v - mean) / std for k, v in values.items()}
    elif method == "robust":
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            normalized = {k: 0.0 for k in values.keys()}
        else:
            normalized = {k: (v - q1) / iqr for k, v in values.items()}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def create_evidence_table(
    statement: str,
    method: str,
    features: List[str],
    contributions: List[float],
    fidelity_score: float
) -> pd.DataFrame:
    """
    Create evidence table linking text to XAI values.
    
    Args:
        statement: natural language statement
        method: XAI method used (SHAP/LIME)
        features: list of feature names
        contributions: list of contribution values
        fidelity_score: alignment score between text and values
        
    Returns:
        DataFrame with evidence traceability
    """
    return pd.DataFrame({
        "statement": [statement] * len(features),
        "method": [method] * len(features),
        "feature": features,
        "contribution": contributions,
        "fidelity_score": [fidelity_score] * len(features)
    })

def log_step(step_name: str, details: Dict[str, Any], verbose: bool = True):
    # log execution steps
    if verbose:
        logger.info(f"[x] {step_name}")
        for key, value in details.items():
            logger.info(f"  {key}: {value}")