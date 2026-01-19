"""
Configuration classes for XAI-NLG Framework.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExplainerConfig:
    """Configuration for XAI explainers (SHAP/LIME)."""
    # SHAP settings
    shap_n_samples: int = 100  # Background samples for KernelExplainer
    shap_model_type: str = "auto"  # "tree", "kernel", or "auto"
    
    # LIME settings
    lime_n_samples: int = 1000  # Number of samples for LIME
    lime_kernel_width: float = 0.75
    
    # General
    top_k_features: int = 5  # Number of top features to show


@dataclass
class NormalizerConfig:
    """Configuration for normalizer and mapper."""
    scale_method: str = "minmax"  # "minmax", "standard", "robust"
    feature_grouping_threshold: float = 0.05  # Group features below this
    

@dataclass
class NLGConfig:
    """Configuration for NLG generators."""
    model_name: str = "llama3:latest"  # Ollama model name
    temperature: float = 0.3
    max_tokens: int = 300
    techniques: List[str] = field(default_factory=lambda: ["few_shot", "cot", "self_consistency"])
    api_key: Optional[str] = None
    debug_print_prompt: bool = False
    

@dataclass
class ValidatorConfig:
    """Configuration for validation."""
    verify_sum_conservation: bool = True
    sum_tolerance: float = 0.1  # Tolerance for SHAP sum check
    min_clarity_score: float = 40.0
    track_evidence: bool = True


@dataclass 
class FrameworkConfig:
    """Main framework configuration."""
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    nlg: NLGConfig = field(default_factory=NLGConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    random_seed: int = 42
    verbose: bool = True