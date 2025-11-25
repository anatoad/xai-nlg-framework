from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional
from config.settings import NLGConfig

LLMCallFn = Callable[[str, NLGConfig], str]

class BaseNLGGenerator(ABC):

    def __init__(self, config: NLGConfig, llm_call_fn: Optional[LLMCallFn] = None):
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.llm_call_fn = llm_call_fn

    @abstractmethod
    def generate(self, context: Dict) -> str:
        """
        Generate natural language explanation.

        Args:
            context: Dictionary with explanation context

        Returns:
            Generated text
        """
        raise NotImplementedError

    def build_prompt(self, context: Dict) -> str:
        """
        Build a simple prompt from the provided context.
        """
        features = context.get("features", [])
        values = context.get("values", [])
        prediction = context.get("prediction", "")

        prompt = f"Explain why the model predicted {prediction}.\n\n"
        prompt += "Key contributing factors:\n"
        for feature, value in zip(features, values):
            prompt += f"- {feature}: {value:.3f}\n"

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Helper: folosește funcția de LLM injectată în constructor.
        """
        if self.llm_call_fn is None:
            raise RuntimeError(
                "No LLM call function provided to BaseNLGGenerator. "
                "Pass llm_call_fn=... when constructing the generator."
            )
        return self.llm_call_fn(prompt, self.config)
