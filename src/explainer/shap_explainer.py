import numpy as np
import shap
import logging
import pandas as pd
from typing import Dict, List
from .base import BaseExplainer

logger = logging.getLogger(__name__)

class SHAPExplainer(BaseExplainer):
    def __init__(self, model, data, feature_names: list, model_type: str = "auto", verbose: bool = False):
        super().__init__(model, data, feature_names)
        self.model_type = model_type  # "tree", "kernel", or "auto"
        self.verbose = verbose
        self._init_explainer()
    
    def _init_explainer(self):
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(self.data, 100)
            )
        elif self.model_type == "auto":
            try:
                self.explainer = shap.TreeExplainer(self.model)
                if self.verbose:
                    logger.info("Using TreeExplainer")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer")
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.data, 100)
                )
    
    def _process_shap_values(self, shap_values, instance_idx: int) -> np.ndarray:
        """
        Process SHAP values to handle different output shapes.
        
        SHAP can return:
        - List[array]: Multiclass or binary case
        - 3D array (n, features, classes): TreeExplainer binary classification
        - 2D array (n, features): Standard case
        
        Args:
            shap_values: Raw SHAP values from explainer
            instance_idx: Index of instance to extract
            
        Returns:
            1D array of shape (n_features,)
        """
        original_shape = None
        
        # log original shape for debugging
        if isinstance(shap_values, list):
            original_shape = f"List of {len(shap_values)} arrays, shapes: {[v.shape for v in shap_values]}"
            if self.verbose:
                logger.debug(f"SHAP values (list): {original_shape}")
            
            # for binary classification, use positive class (index 1)
            shap_values = shap_values if len(shap_values) > 1 else shap_values
        
        if isinstance(shap_values, np.ndarray):
            original_shape = shap_values.shape
            if self.verbose:
                logger.debug(f"SHAP values array shape: {original_shape}")
        
        # extract values based on dimensionality
        if shap_values.ndim == 3:
            # shape: (n_samples, n_features, 2) - TreeExplainer binary classification
            if self.verbose:
                logger.debug(f"Handling 3D array: extracting positive class (dim 2, index 1)")
            result = shap_values[instance_idx, :, 1]
        elif shap_values.ndim == 2:
            # shape: (n_samples, n_features) - standard case
            result = shap_values[instance_idx, :]
        elif shap_values.ndim == 1:
            # shape: (n_features,) - single instance already
            result = shap_values
        else:
            raise ValueError(
                f"Unexpected SHAP values dimensionality: {shap_values.ndim}. "
                f"Original shape info: {original_shape}"
            )
        
        if self.verbose:
            logger.debug(f"Final extracted values shape: {result.shape}")
        
        return result
    
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        if self.verbose:
            logger.info(f"Generating SHAP explanation for instance of shape {instance.shape}")
        
        instance_2d = instance.reshape(1, -1)
        shap_values = self.explainer.shap_values(instance_2d)
        
        # Process and reshape SHAP values
        processed_values = self._process_shap_values(shap_values, 0)
        
        if self.verbose:
            logger.debug(f"Processed values shape: {processed_values.shape}")
        
        explanation = {
            self.feature_names[i]: float(processed_values[i])
            for i in range(self.n_features)
        }
        
        if self.verbose:
            top_features = sorted(explanation.items(), key=lambda x: abs(x), reverse=True)[:3]
            logger.info(f"Top 3 features: {top_features}")
        
        return explanation
    
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        if self.verbose:
            logger.info(f"Generating SHAP explanations for {len(instances)} instances")
        
        shap_values = self.explainer.shap_values(instances)
        
        explanations = []
        for i in range(len(instances)):
            processed_values = self._process_shap_values(shap_values, i)
            explanation = {
                self.feature_names[j]: float(processed_values[j])
                for j in range(self.n_features)
            }
            explanations.append(explanation)
        
        if self.verbose:
            logger.info(f"Successfully generated {len(explanations)} explanations")
        
        return explanations
    
    def get_base_value(self) -> float:
        base = self.explainer.expected_value
        
        if self.verbose:
            logger.debug(f"Base value type: {type(base)}, value: {base}")
        
        # handle list of base values (multiclass)
        if isinstance(base, (list, np.ndarray)):
            if len(base) == 1:
                return float(base[0])
            else:
                if self.verbose:
                    logger.debug(f"Multiple base values detected, selecting index 1: {base[1]}")
                return float(base[1])  # choose positive class (index 1)

        return float(base)

    def explanation_to_dataframe(
            self,
            explanation: Dict[str, float],
            instance: np.ndarray = None,
            sort_by: str = "absolute"
        ) -> pd.DataFrame:
        return self._explanation_to_dataframe("SHAP", explanation=explanation, instance=instance, sort_by=sort_by)
