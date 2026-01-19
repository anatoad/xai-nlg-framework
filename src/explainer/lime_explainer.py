"""
LIME Explainer implementation.

LIME (Local Interpretable Model-agnostic Explanations) explains predictions
by approximating the model locally with an interpretable model.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME-based explainer for tabular data.
    
    LIME generates explanations by:
    1. Sampling perturbations around the instance
    2. Getting model predictions for perturbations
    3. Fitting a local linear model weighted by proximity
    4. Using linear model coefficients as feature importance
    """
    
    def __init__(
        self,
        model,
        data: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        n_samples: int = 1000,
        kernel_width: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained ML model with predict_proba method
            data: Training data for statistics
            feature_names: List of feature names
            categorical_features: Indices of categorical features
            n_samples: Number of samples for local approximation
            kernel_width: Width of the kernel (default: 0.75 * sqrt(n_features))
            verbose: Whether to print debug information
        """
        super().__init__(model, data, feature_names)
        self.categorical_features = categorical_features or []
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.verbose = verbose
        self.explainer = None
        
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize the LIME TabularExplainer."""
        try:
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "LIME library is required. Install with: pip install lime"
            )
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.data,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features if self.categorical_features else None,
            verbose=self.verbose,
            mode='classification',
            random_state=42
        )
        
        if self.verbose:
            logger.info(f"LIME explainer initialized with {self.n_samples} samples")
    
    def _get_predict_fn(self):
        """Get the prediction function for LIME."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        else:
            # Wrap predict to return probabilities-like output
            def predict_wrapper(X):
                preds = self.model.predict(X)
                # Convert to probability-like format
                proba = np.zeros((len(preds), 2))
                proba[:, 1] = preds
                proba[:, 0] = 1 - preds
                return proba
            return predict_wrapper
    
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Generate LIME explanation for a single instance.

        Args:
            instance: 1D array of feature values

        Returns:
            Dictionary mapping feature names to LIME weights
        """
        # Ensure 1D
        instance = instance.flatten()

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            instance,
            self._get_predict_fn(),
            num_features=self.n_features,
            num_samples=self.n_samples
        )

        # Extract feature contributions using as_list() which returns [(feature_name, weight), ...]
        exp_list = exp.as_list()
        
        # DEBUG
        print("DEBUG exp.as_list():", exp_list[:5])  # primele 5
        
        # Build explanation dictionary - initialize with 0
        explanation = {name: 0.0 for name in self.feature_names}
        
        # LIME as_list() returns tuples like ('feature_name <= value', weight) or ('value < feature_name', weight)
        # We need to extract the feature name from the description
        for description, weight in exp_list:
            # Find which feature this description refers to
            for feature_name in self.feature_names:
                if feature_name in description:
                    explanation[feature_name] = float(weight)
                    break
        
        if self.verbose:
            top_3 = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            logger.info(f"Top 3 LIME features: {top_3}")
        
        return explanation
    
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        """
        Generate LIME explanations for multiple instances.
        
        Args:
            instances: 2D array of shape (n_samples, n_features)
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for i in range(len(instances)):
            explanation = self.explain(instances[i])
            explanations.append(explanation)
            
            if self.verbose and (i + 1) % 10 == 0:
                logger.info(f"LIME: Explained {i + 1}/{len(instances)} instances")
        
        return explanations
    
    def get_base_value(self) -> float:
        """
        LIME doesn't have a base value like SHAP.
        Returns 0.0 as placeholder.
        """
        return 0.0