from typing import Dict, Any, Optional
import numpy as np

from config.settings import FrameworkConfig
from src.explainer.shap_explainer import SHAPExplainer
from src.explainer.lime_explainer import LIMEExplainer
from src.normalizer.normalizer import Normalizer
from src.normalizer.mapper import FeatureMapper
from src.nlg.few_shot_generator import FewShotGenerator
from src.nlg.cot_generator import ChainOfThoughtGenerator
from src.nlg.self_consistency_generator import SelfConsistencyGenerator
from src.nlg.ollama_client import ollama_llm_call
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
        config: Optional[FrameworkConfig] = None,
    ):
        """
        Initialize pipeline.

        Args:
            model: Trained ML model (must implement .predict and optionally .predict_proba)
            data: Training data (for SHAP/LIME background)
            feature_names: Feature names in order
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

        # layer 3: NLG generators (Few-Shot, CoT, Self-Consistency)
        # All of them use the same NLGConfig and the same Ollama client wrapper.
        self.few_shot_generator = FewShotGenerator(
            self.config.nlg,
            llm_call_fn=ollama_llm_call,
        )
        self.cot_generator = ChainOfThoughtGenerator(
            self.config.nlg,
            llm_call_fn=ollama_llm_call,
        )
        self.sc_generator = SelfConsistencyGenerator(
            self.config.nlg,
            n_chains=3,
            llm_call_fn=ollama_llm_call,
        )

        # layer 4: validator & evidence tracker
        self.validator = XAIValidator(self.config.validator)
        self.evidence_tracker = EvidenceTracker()

    def explain_instance(
        self,
        instance: np.ndarray,
        method: str = "shap",
        generate_text: bool = True,
        technique: Optional[str] = None,
        audience: str = "expert",
    ) -> Dict[str, Any]:
        """
        Explain a single instance through the full pipeline.

        Args:
            instance: 1D NumPy array with feature values
            method: "shap" or "lime"
            generate_text: Whether to generate natural language explanation
            technique:
                - "few_shot" (default)
                - "cot"
                - "self_consistency"
            audience:
                - "expert"  -> technical explanation (for Few-Shot)
                - "layman"  -> simple explanation (for Few-Shot)
                - "both"    -> expert + layman in one text (only Few-Shot supports this explicitly)

        Returns:
            Dictionary with:
              - instance, method
              - explanation (raw feature->value dict)
              - base_value (SHAP expected value or 0.0 for LIME)
              - prediction (raw model prediction)
              - ranked_features (list of (feature, contribution))
              - statements (mapped feature statements, if mapper supports it)
              - generated_text (if generate_text=True)
              - validation (clarity score, sum conservation, etc.)
              - nlg_technique, nlg_generator (metadata)
        """
        # ------------------ Layer 1: Explain ------------------ #
        log_step(
            "Layer 1: Generating explanations",
            {"method": method},
            self.config.verbose,
        )

        if method.lower() == "shap":
            explanation = self.shap_explainer.explain(instance)
            base_value = self.shap_explainer.get_base_value()
        elif method.lower() == "lime":
            explanation = self.lime_explainer.explain(instance)
            base_value = 0.0
        else:
            raise ValueError(f"Unsupported explanation method: {method}")

        # model prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]

        # ------------------ Layer 2: Normalize & Map ------------------ #
        log_step(
            "Layer 2: Normalizing",
            {"features": len(explanation)},
            self.config.verbose,
        )

        normalized = self.normalizer.normalize_contributions(explanation)
        ranked = self.normalizer.rank_features(explanation)
        statements = self.mapper.map_features(ranked, normalized)

        result: Dict[str, Any] = {
            "instance": instance,
            "method": method,
            "explanation": explanation,
            "base_value": base_value,
            "prediction": prediction,
            "ranked_features": ranked,
            "statements": statements,
        }

        # ------------------ Layer 3: NLG (optional) ------------------ #
        if generate_text:
            # choose technique
            tech = (technique or "").lower()
            if not tech:
                # default: first configured technique or 'few_shot'
                tech = (
                    self.config.nlg.techniques[0]
                    if getattr(self.config.nlg, "techniques", None)
                    else "few_shot"
                )
            tech = tech.lower()

            if tech == "cot":
                generator = self.cot_generator
                generator_name = "cot"
            elif tech in (
                "self_consistency",
                "self-consistency",
                "selfconsistency",
                "sc",
            ):
                generator = self.sc_generator
                generator_name = "self_consistency"
            else:
                # fallback to Few-Shot
                generator = self.few_shot_generator
                generator_name = "few_shot"
                tech = "few_shot"

            log_step(
                "Layer 3: Generating text",
                {"generator": generator_name, "technique": tech, "audience": audience},
                self.config.verbose,
            )

            # build context for NLG (top-k features)
            k = self.config.explainer.top_k_features
            top_k = ranked[:k]

            directions = []
            for _, v in top_k:
                if v > 0:
                    directions.append("supports")
                elif v < 0:
                    directions.append("contradicts")
                else:
                    directions.append("neutral")

            context = {
                "features": [f for f, _ in top_k],
                "values": [float(v) for _, v in top_k],
                "directions": directions,
                "method": method,
                # generic stringified prediction; for a specific dataset you can map this
                # to domain-specific labels (e.g. 'benign tumor', 'malignant tumor').
                "prediction": str(prediction),
            }

            # audience is mainly used by FewShotGenerator
            if tech == "few_shot":
                context["audience"] = audience

            generated_text = generator.generate(context)

            # ------------------ Layer 4: Validation ------------------ #
            log_step(
                "Layer 4: Validating",
                {},
                self.config.verbose,
            )

            validation = self.validator.validate_all(
                explanation=explanation,
                generated_text=generated_text,
                base_value=base_value,
                prediction=float(prediction)
                if isinstance(prediction, (float, int, np.floating))
                else 0.0,
                method=method.lower(),
            )

            result["generated_text"] = generated_text
            result["validation"] = validation
            result["nlg_technique"] = tech
            result["nlg_generator"] = generator_name

        return result
