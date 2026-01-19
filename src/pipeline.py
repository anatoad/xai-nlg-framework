"""
XAI-NLG Pipeline

Main pipeline that combines all components:
1. Explainer (SHAP/LIME) - generates feature contributions
2. Normalizer & Mapper - processes and converts to statements
3. NLG Generator - produces natural language explanations
4. Validator - validates the explanations
"""
import numpy as np
from typing import Dict, Any, Optional, List

from config.settings import FrameworkConfig
from src.explainer.shap_explainer import SHAPExplainer
from src.explainer.lime_explainer import LIMEExplainer
from src.normalizer.normalizer import Normalizer
from src.normalizer.mapper import FeatureMapper
from src.nlg.few_shot_generator import FewShotGenerator
from src.nlg.cot_generator import ChainOfThoughtGenerator
from src.nlg.self_consistency_generator import SelfConsistencyGenerator
from src.validator.validator import ExplanationValidator
from src.validator.evidence_tracker import EvidenceTracker


class XAINLGPipeline:
    """
    Main pipeline for transforming XAI explanations into natural language.
    
    Usage:
        pipeline = XAINLGPipeline(model, X_train, feature_names)
        result = pipeline.explain_instance(X_test[0], method="shap")
        print(result["generated_text"])
    """
    
    def __init__(
        self,
        model,
        data: np.ndarray,
        feature_names: List[str],
        config: Optional[FrameworkConfig] = None,
        llm_call_fn=None
    ):
        """
        Initialize the XAI-NLG pipeline.
        
        Args:
            model: Trained ML model with predict/predict_proba methods
            data: Training data for background (numpy array)
            feature_names: List of feature names
            config: Framework configuration
            llm_call_fn: Optional function to call LLM (prompt, config) -> str
        """
        self.model = model
        self.data = np.array(data)
        self.feature_names = list(feature_names)
        self.config = config or FrameworkConfig()
        self.llm_call_fn = llm_call_fn
        
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Layer 1: Explainers
        self._shap_explainer = None  # Lazy initialization
        self._lime_explainer = None  # Lazy initialization
        
        # Layer 2: Normalizer & Mapper
        self.normalizer = Normalizer(self.config.normalizer)
        self.mapper = FeatureMapper()
        
        # Layer 3: NLG Generators
        self.generators = {
            "few_shot": FewShotGenerator(self.config.nlg, self.llm_call_fn),
            "cot": ChainOfThoughtGenerator(self.config.nlg, self.llm_call_fn),
            "self_consistency": SelfConsistencyGenerator(self.config.nlg, self.llm_call_fn)
        }
        
        # Layer 4: Validator & Evidence Tracker
        self.validator = ExplanationValidator(self.config.validator)
        self.evidence_tracker = EvidenceTracker()
    
    @property
    def shap_explainer(self) -> SHAPExplainer:
        """Lazy initialization of SHAP explainer."""
        if self._shap_explainer is None:
            self._shap_explainer = SHAPExplainer(
                self.model,
                self.data,
                self.feature_names,
                model_type=self.config.explainer.shap_model_type,
                n_background_samples=self.config.explainer.shap_n_samples,
                verbose=self.config.verbose
            )
        return self._shap_explainer
    
    @property
    def lime_explainer(self) -> LIMEExplainer:
        """Lazy initialization of LIME explainer."""
        if self._lime_explainer is None:
            self._lime_explainer = LIMEExplainer(
                self.model,
                self.data,
                self.feature_names,
                n_samples=self.config.explainer.lime_n_samples,
                verbose=self.config.verbose
            )
        return self._lime_explainer
    
    def explain_instance(
        self,
        instance: np.ndarray,
        method: str = "shap",
        technique: str = "few_shot",
        audience: str = "expert",
        generate_text: bool = True,
        instance_id: str = None,
        track_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single instance.
        
        Args:
            instance: 1D numpy array of feature values
            method: XAI method - "shap" or "lime"
            technique: NLG technique - "few_shot", "cot", or "self_consistency"
            audience: Target audience - "expert", "layman", or "both"
            generate_text: Whether to generate NLG text
            instance_id: Optional instance identifier for tracking
            track_evidence: Whether to track in evidence log
            
        Returns:
            Dictionary containing:
                - prediction: Model prediction
                - method: XAI method used
                - contributions: Raw feature contributions
                - ranked_features: Features sorted by importance
                - statements: FeatureStatement objects
                - generated_text: NLG explanation (if generate_text=True)
                - validation: Validation results
        """
        instance = np.array(instance).flatten()
        
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Explaining instance with {method.upper()}")
            print(f"{'='*50}")
        
        # ============ LAYER 1: XAI Explanation ============
        if self.config.verbose:
            print("\n[Layer 1] Generating XAI explanation...")
        
        if method.lower() == "shap":
            contributions = self.shap_explainer.explain(instance)
            base_value = self.shap_explainer.get_base_value()
        elif method.lower() == "lime":
            contributions = self.lime_explainer.explain(instance)
            base_value = 0.0
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap' or 'lime'.")
        
        # Get model prediction
        pred = self.model.predict(instance.reshape(1, -1))[0]
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance.reshape(1, -1))[0]
            pred_proba = proba[1] if len(proba) == 2 else max(proba)
        else:
            pred_proba = float(pred)
        
        # ============ LAYER 2: Normalize & Map ============
        if self.config.verbose:
            print("[Layer 2] Normalizing and mapping features...")
        
        normalized = self.normalizer.normalize(contributions)
        ranked_features = self.normalizer.rank_features(
            contributions, 
            top_k=self.config.explainer.top_k_features
        )
        statements = self.mapper.map_features(ranked_features, normalized)
        
        # Build result
        result = {
            "instance": instance,
            "prediction": int(pred),
            "prediction_proba": pred_proba,
            "method": method.upper(),
            "base_value": base_value,
            "contributions": contributions,
            "normalized": normalized,
            "ranked_features": ranked_features,
            "statements": statements
        }
        
        if self.config.verbose:
            print(f"  Top {len(ranked_features)} features:")
            for feat, val in ranked_features[:5]:
                print(f"    - {feat}: {val:.4f}")
        
        # ============ LAYER 3: NLG Generation ============
        if generate_text:
            if self.config.verbose:
                print(f"[Layer 3] Generating text with {technique}...")
            
            # Build context for NLG
            context = self.mapper.get_summary_context(
                statements,
                prediction=str(pred),
                method=method.upper()
            )
            context["audience"] = audience
            
            # Get generator
            generator = self.generators.get(technique, self.generators["few_shot"])
            
            # Generate text
            generated_text = generator.generate(context)
            result["generated_text"] = generated_text
            result["nlg_technique"] = technique
            
            if self.config.verbose:
                print(f"  Generated text ({len(generated_text)} chars)")
            
            # ============ LAYER 4: Validation ============
            if self.config.verbose:
                print("[Layer 4] Validating explanation...")
            
            validation = self.validator.validate_explanation(
                contributions=contributions,
                generated_text=generated_text,
                base_value=base_value,
                prediction=pred_proba,
                method=method
            )
            result["validation"] = validation
            
            if self.config.verbose:
                print(f"  Clarity score: {validation['clarity']['score']:.1f}")
                print(f"  Coverage score: {validation['coverage']['coverage_score']:.2f}")
                print(f"  Valid: {validation['valid']}")
            
            # Track evidence
            if track_evidence:
                self.evidence_tracker.add_record(
                    instance_id=instance_id or f"instance_{len(self.evidence_tracker.records)}",
                    method=method,
                    prediction=str(pred),
                    contributions=contributions,
                    generated_text=generated_text,
                    nlg_technique=technique,
                    validation_results=validation
                )
        
        return result
    
    def explain_batch(
        self,
        instances: np.ndarray,
        method: str = "shap",
        technique: str = "few_shot",
        audience: str = "expert"
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple instances.
        
        Args:
            instances: 2D numpy array (n_samples, n_features)
            method: XAI method
            technique: NLG technique
            audience: Target audience
            
        Returns:
            List of result dictionaries
        """
        results = []
        for i, instance in enumerate(instances):
            if self.config.verbose:
                print(f"\nProcessing instance {i+1}/{len(instances)}")
            
            result = self.explain_instance(
                instance,
                method=method,
                technique=technique,
                audience=audience,
                instance_id=f"batch_{i}"
            )
            results.append(result)
        
        return results
    
    def compare_methods(
        self,
        instance: np.ndarray,
        technique: str = "few_shot"
    ) -> Dict[str, Any]:
        """
        Compare SHAP and LIME explanations for the same instance.
        
        Args:
            instance: 1D numpy array
            technique: NLG technique to use
            
        Returns:
            Dictionary with both explanations and comparison
        """
        shap_result = self.explain_instance(
            instance, method="shap", technique=technique, track_evidence=False
        )
        lime_result = self.explain_instance(
            instance, method="lime", technique=technique, track_evidence=False
        )
        
        # Compare top features
        shap_top = set(f for f, _ in shap_result["ranked_features"][:5])
        lime_top = set(f for f, _ in lime_result["ranked_features"][:5])
        
        overlap = shap_top & lime_top
        jaccard = len(overlap) / len(shap_top | lime_top) if (shap_top | lime_top) else 0
        
        return {
            "shap": shap_result,
            "lime": lime_result,
            "comparison": {
                "shap_top_5": list(shap_top),
                "lime_top_5": list(lime_top),
                "overlap": list(overlap),
                "jaccard_similarity": jaccard
            }
        }