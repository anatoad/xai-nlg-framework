import numpy as np
import pandas as pd
import lime.lime_tabular
from typing import Dict, List, Optional
from .base import BaseExplainer

class LIMEExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        data,
        feature_names: list,
        categorical_features: Optional[list] = None,
        n_samples: int = 1000
    ):
        super().__init__(model, data, feature_names)
        self.categorical_features = categorical_features or []
        self.n_samples = n_samples
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            verbose=False,
            random_state=42
        )
    
    def _extract_feature_name_from_description(self, description: str) -> str:
        """
        Extract clean feature name from LIME's description
        LIME returns descriptions like:
        - "516.45 < worst area" -> "worst area"
        - "84.25 < worst perimeter" -> "worst perimeter"
        - "malignant = True" -> "malignant"
        
        Args:
            description: LIME's feature description
            
        Returns:
            Clean feature name
        """
        # handle numerical comparisons (e.g., "516.45 < worst area")
        if "<" in description or ">" in description or "≤" in description or "≥" in description:
            # split by comparison operator and take the part after it
            for op in ["<", ">", "≤", "≥", "<=", ">="]:
                if op in description:
                    parts = description.split(op)
                    if len(parts) >= 2:
                        feature_name = parts[1].strip()
                        return feature_name
        
        # handle categorical values (e.g., "feature = value")
        if "=" in description:
            parts = description.split("=")
            if len(parts) == 2:
                return parts[0].strip()
        
        return description.strip()

    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """Generate LIME explanation for a single instance."""
        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=self.n_features,
            num_samples=self.n_samples,
        )

        # Use index-based map instead of human-readable strings
        exp_map = exp.as_map()
        # exp_map is {class_index: [(feature_idx, weight), ...]}

        if isinstance(exp_map, dict):
            if len(exp_map) == 1:
                # single output (regression or single-class map)
                contribs = next(iter(exp_map.values()))
            else:
                # binary classifier: use positive class 1 by convention
                contribs = exp_map.get(1, next(iter(exp_map.values())))
        else:
            contribs = exp_map

        explanation: Dict[str, float] = {}

        # Aggregate contributions per original feature
        for idx, weight in contribs:
            fname = str(self.feature_names[idx])
            explanation[fname] = explanation.get(fname, 0.0) + float(weight)

        # Ensure all features are present
        for fname in self.feature_names:
            explanation.setdefault(fname, 0.0)

        return explanation

    
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        return [self.explain(instance) for instance in instances]

    def explanation_to_dataframe(
        self,
        explanation: Dict[str, float],
        instance: np.ndarray = None,
        sort_by: str = "absolute"
    ) -> pd.DataFrame:
        return self._explanation_to_dataframe("LIME", explanation=explanation, instance=instance, sort_by=sort_by)
