from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd

class BaseExplainer(ABC):
    def __init__(self, model, data, feature_names: list):
        """
        Initialize the explainer with model
        Args:
            model: trained model with predict method
            data: training data (numpy array or DataFrame)
            feature_names: list of feature names
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.n_features = len(feature_names)
    
    @abstractmethod
    def explain(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Generate explanation for a single instance        
        Returns:
            Dictionary mapping feature names to contribution values
        """
        pass
    
    @abstractmethod
    def explain_batch(self, instances: np.ndarray) -> list:
        """ Generate explanations for a batch """
        pass

    @abstractmethod
    def explanation_to_dataframe(
        self,
        explanation: Dict[str, float],
        instance: np.ndarray = None,
        sort_by: str = "absolute"
    ) -> pd.DataFrame:
        """
        Convert explanation to DataFrame for tabular visualization.
        
        Args:
            explanation: Dictionary of feature names to explainer weights
            instance: Original instance values (optional, for feature display)
            sort_by: "absolute" (default), "positive", "negative", or None
            
        Returns:
            DataFrame with columns: Feature, Explainer Weight, Impact, Feature Value (if instance provided)
        """
        pass
    
    def get_explanation_metadata(self, explanation: Dict[str, float]) -> Dict[str, Any]:
        """ extract metadata about explanation """
        values = list(explanation.values())
        return {
            "sum": sum(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
            "positive_count": sum(1 for v in values if v > 0),
            "negative_count": sum(1 for v in values if v < 0),
        }

    def _explanation_to_dataframe(
        self,
        explainer_name: str,
        explanation: Dict[str, float],
        instance: np.ndarray = None,
        sort_by: str = "absolute",
    ) -> pd.DataFrame:
            data = []
            
            for feature_name, shap_value in explanation.items():
                # get feature value if instance provided
                feature_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else -1
                feature_value = instance[feature_idx] if instance is not None and feature_idx >= 0 else None
                
                # determine impact direction
                if shap_value > 0:
                    impact = "Positive"
                elif shap_value < 0:
                    impact = "Negative"
                else:
                    impact = "Neutral"
                
                row = {
                    "Feature": feature_name,
                    f"{explainer_name} Value": shap_value,
                    "Absolute Value": abs(shap_value),
                    "Impact": impact,
                }

                if feature_value is not None:
                    row["Feature Value"] = feature_value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            if sort_by == "absolute":
                df = df.sort_values("Absolute Value", ascending=False)
            elif sort_by == "positive":
                df = df.sort_values(f"{explainer_name} Value", ascending=False)
            elif sort_by == "negative":
                df = df.sort_values(f"{explainer_name} Value", ascending=True)
            
            # drop helper column if not needed
            if "Feature Value" not in df.columns or df["Feature Value"].isna().all():
                df = df.drop("Absolute Value", axis=1)
            else:
                df = df[["Feature", "Feature Value", f"{explainer_name} Value", "Absolute Value", "Impact"]]
            
            return df.reset_index(drop=True)
