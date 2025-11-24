import numpy as np
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
    
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=self.n_features,
            num_samples=self.n_samples
        )
        
        explanation = {}
        for feature_idx, weight in exp.as_list():
            # extract feature name and weight
            if "<=" in feature_idx or ">" in feature_idx:
                feature_name = feature_idx.split("<=")[0].split(">")[0].strip()
                explanation[feature_name] = float(weight)
            else:
                explanation[feature_idx] = float(weight)
        
        # fill missing features with 0
        for fname in self.feature_names:
            if fname not in explanation:
                explanation[fname] = 0.0

        return explanation
    
    def explain_batch(self, instances: np.ndarray) -> List[Dict[str, float]]:
        return [self.explain(instance) for instance in instances]

