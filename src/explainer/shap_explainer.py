"""
SHAP Explainer implementation.

SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values
to explain individual predictions.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExplainer

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainer using TreeExplainer or KernelExplainer.
    
    For tree-based models (Random Forest, XGBoost, etc.), TreeExplainer
    is used for exact Shapley values. For other models, KernelExplainer
    provides model-agnostic approximations.
    """
    
    def __init__(
        self,
        model,
        data: np.ndarray,
        feature_names: List[str],
        model_type: str = "auto",
        n_background_samples: int = 100,
        verbose: bool = False
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained ML model
            data: Training data for background distribution
            feature_names: List of feature names
            model_type: "tree", "kernel", or "auto" (auto-detect)
            n_background_samples: Number of background samples for KernelExplainer
            verbose: Whether to print debug information
        """
        super().__init__(model, data, feature_names)
        self.model_type = model_type
        self.n_background_samples = n_background_samples
        self.verbose = verbose
        self.explainer = None
        self._base_value = None
        
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize the appropriate SHAP explainer."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP library is required. Install with: pip install shap"
            )
        
        if self.model_type == "tree":
            self._init_tree_explainer(shap)
        elif self.model_type == "kernel":
            self._init_kernel_explainer(shap)
        elif self.model_type == "auto":
            # Try TreeExplainer first, fall back to KernelExplainer
            try:
                self._init_tree_explainer(shap)
                if self.verbose:
                    logger.info("Using TreeExplainer (auto-detected tree model)")
            except Exception as e:
                if self.verbose:
                    logger.info(f"TreeExplainer failed ({e}), using KernelExplainer")
                self._init_kernel_explainer(shap)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _init_tree_explainer(self, shap):
        """Initialize TreeExplainer for tree-based models."""
        self.explainer = shap.TreeExplainer(self.model)
        self._explainer_type = "tree"
        
    def _init_kernel_explainer(self, shap):
        """Initialize KernelExplainer for model-agnostic explanations."""
        # Sample background data
        if len(self.data) > self.n_background_samples:
            background = shap.sample(self.data, self.n_background_samples)
        else:
            background = self.data
            
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba if hasattr(self.model, 'predict_proba') 
            else self.model.predict,
            background
        )
        self._explainer_type = "kernel"
    
    def _extract_shap_values(self, shap_values, instance_idx: int = 0) -> np.ndarray:
        """
        Extract SHAP values handling different output formats.
        
        SHAP can return values in various formats:
        - List of arrays: [array_class0, array_class1] for binary classification
        - 3D array: (n_samples, n_features, n_classes) for TreeExplainer
        - 2D array: (n_samples, n_features) for regression/single output
        
        For classification, we use the positive class (index 1).
        
        Args:
            shap_values: Raw SHAP values from explainer
            instance_idx: Index of the instance to extract
            
        Returns:
            1D array of shape (n_features,)
        """
        # Handle list format (binary classification)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # Binary classification: use positive class
                shap_values = shap_values[1]
            else:
                # Multi-class: use the predicted class or first one
                shap_values = shap_values[0]
        
        # Convert to numpy array if needed
        shap_values = np.array(shap_values)
        
        # Handle different dimensionalities
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) - use positive class
            values = shap_values[instance_idx, :, 1] if shap_values.shape[2] == 2 else shap_values[instance_idx, :, 0]
        elif shap_values.ndim == 2:
            # (n_samples, n_features)
            values = shap_values[instance_idx, :]
        elif shap_values.ndim == 1:
            # Already 1D (single instance)
            values = shap_values
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
        
        return values.flatten()
    
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance: 1D array of feature values
            
        Returns:
            Dictionary mapping feature names to SHAP values
        """
        # Ensure 2D shape for SHAP
        instance_2d = instance.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance_2d)
        
        # Extract values for this instance
        values = self._extract_shap_values(shap_values, 0)
        
        # Create explanation dictionary
        explanation = {
            self.feature_names[i]: float(values[i])
            for i in range(self.n_features)
        }
        
        if self.verbose:
            top_3 = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            logger.info(f"Top 3 SHAP features: {top_3}")
        
        return explanation
    
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        """
        Generate SHAP explanations for multiple instances.
        
        Args:
            instances: 2D array of shape (n_samples, n_features)
            
        Returns:
            List of explanation dictionaries
        """
        shap_values = self.explainer.shap_values(instances)
        
        explanations = []
        for i in range(len(instances)):
            values = self._extract_shap_values(shap_values, i)
            explanation = {
                self.feature_names[j]: float(values[j])
                for j in range(self.n_features)
            }
            explanations.append(explanation)
        
        return explanations
    
    def get_base_value(self) -> float:
        """
        Get the expected value (base value) from the explainer.
        
        For TreeExplainer, this is the mean prediction over the training data.
        """
        base = self.explainer.expected_value
        
        # Handle array/list of base values (multi-class)
        if isinstance(base, (list, np.ndarray)):
            base = np.array(base)
            if base.ndim == 0:
                return float(base)
            elif len(base) == 2:
                # Binary classification: return positive class base value
                return float(base[1])
            else:
                return float(base[0])
        
        return float(base)