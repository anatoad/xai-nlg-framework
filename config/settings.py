from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExplainerConfig:
    shap_n_samples: int = 100
    lime_n_samples: int = 1000
    lime_feature_selection: str = "auto"
    top_k_features: int = 5

@dataclass
class NormalizerConfig:
    scale_method: str = "minmax"  # minmax, standard, robust
    feature_grouping_threshold: float = 0.1
    text_template_style: str = "concise"  # concise, detailed, educational

@dataclass
class NLGConfig:
    model_name: str = "llama3:latest"
    # model_name: str = "qwen3-vl:235b"
    # model_name: str = "deepseek-r1:8b"
    temperature: float = 0.3
    max_tokens: int = 200
    techniques: List[str] = None  # ["few_shot", "cot", "self_consistency"]
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.techniques is None:
            self.techniques = ["few_shot", "cot", "self_consistency"]

@dataclass
class ValidatorConfig:
    verify_sum_conservation: bool = True
    sum_tolerance: float = 1e-5
    track_evidence: bool = True
    min_clarity_score: float = 45.0
    min_robustness_jaccard: float = 0.6

@dataclass
class FrameworkConfig:
    explainer: ExplainerConfig = None
    normalizer: NormalizerConfig = None
    nlg: NLGConfig = None
    validator: ValidatorConfig = None
    random_seed: int = 42
    verbose: bool = False

    def __post_init__(self):
        if self.explainer is None:
            self.explainer = ExplainerConfig()
        if self.normalizer is None:
            self.normalizer = NormalizerConfig()
        if self.nlg is None:
            self.nlg = NLGConfig()
        if self.validator is None:
            self.validator = ValidatorConfig()
