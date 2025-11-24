from typing import Dict, Any, Optional
import numpy as np
from config.settings import FrameworkConfig
from src.explainer.shap_explainer import SHAPExplainer
from src.explainer.lime_explainer import LIMEExplainer
from src.normalizer.normalizer import Normalizer
from src.normalizer.mapper import FeatureMapper
from src.nlg.few_shot_generator import FewShotGenerator
from src.nlg.cot_generator import ChainOfThoughtGenerator
from src.validator.validator import XAIValidator
from src.validator.evidence_tracker import EvidenceTracker
from src.utils import log_step

class XAINLGPipeline:
    """Main pipeline for XAI-to-NLG transformation."""
    
    def __init__(
        self,
        model,
        data,
        feature_names: list,
        config: FrameworkConfig = None
    ):
        """
        Initialize pipeline.
        
        Args:
            model: Trained ML model
            data: Training data
            feature_names: Feature names
            config: Framework configuration
        """
        self.config = config or FrameworkConfig()
        self.model = model
        self.data = data
        self.feature_names = feature_names
        
        self._init_components()
    
    def _init_components(self):
        # layer 1: explainers
        self.shap_explainer = SHAPExplainer(
            self.model, self.data, self.feature_names, model_type="auto"
        )
        self.lime_explainer = LIMEExplainer(
            self.model, self.data, self.feature_names
        )
        
        # layer 2: normalizer & mapper
        self.normalizer = Normalizer(self.config.normalizer)
        self.mapper = FeatureMapper()
        
        # layer 3: NLG
        self.few_shot_generator = FewShotGenerator(self.config.nlg)
        self.cot_generator = ChainOfThoughtGenerator(self.config.nlg)
        
        # layer 4: validator & tracker
        self.validator = XAIValidator(self.config.validator)
        self.evidence_tracker = EvidenceTracker()
    
    def explain_instance(
        self,
        instance: np.ndarray,
        method: str = "shap",
        generate_text: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single instance through full pipeline.
        
        Args:
            instance: Input instance
            method: "shap" or "lime"
            generate_text: Whether to generate natural language
            
        Returns:
            Dictionary with explanations and generated text
        """
        log_step("Layer 1: Generating explanations", {"method": method}, self.config.verbose)
        
        # layer 1: get explanation
        if method == "shap":
            explanation = self.shap_explainer.explain(instance)
            base_value = self.shap_explainer.get_base_value()
        else:
            explanation = self.lime_explainer.explain(instance)
            base_value = 0.0
        
        # get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        log_step("Layer 2: Normalizing", {"features": len(explanation)}, self.config.verbose)
        
        # layer 2: normalize and map
        normalized = self.normalizer.normalize_contributions(explanation)
        ranked = self.normalizer.rank_features(explanation)
        statements = self.mapper.map_features(ranked, normalized)
        
        result = {
            "instance": instance,
            "method": method,
            "explanation": explanation,
            "base_value": base_value,
            "prediction": prediction,
            "ranked_features": ranked,
            "statements": statements
        }
        
        if generate_text:
            log_step("Layer 3: Generating text", {"generator": "few_shot"}, self.config.verbose)
            
            # layer 3: generate text
            context = {
                "features": [f for f, _ in ranked[:5]],
                "values": [v for _, v in ranked[:5]],
                "prediction": str(prediction)
            }
            generated_text = self.few_shot_generator.generate(context)
            
            log_step("Layer 4: Validating", {}, self.config.verbose)
            
            # layer 4: validate
            validation = self.validator.validate_all(
                explanation, generated_text, base_value, prediction
            )
            
            result["generated_text"] = generated_text
            result["validation"] = validation
        
        return result
