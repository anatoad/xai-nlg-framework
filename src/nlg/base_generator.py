from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional
from config.settings import NLGConfig

# Type alias for the low-level LLM call function (e.g. ollama_llm_call)
LLMCallFn = Callable[[str, NLGConfig], str]

class BaseNLGGenerator(ABC):
    """
    Base class for all NLG generators in the framework.

    Subclasses are responsible for:
    - building task-specific prompts from a `context` dict
    - optionally calling an external LLM through `llm_call_fn`
      (for example, an Ollama-backed model)
    """

    def __init__(self, config: NLGConfig, llm_call_fn: Optional[LLMCallFn] = None):
        """
        Args:
            config: NLGConfig with model name, temperature, etc.
            llm_call_fn: function that actually calls the LLM
                         signature: (prompt: str, config: NLGConfig) -> str
        """
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Function that actually calls the LLM (e.g. ollama_llm_call)
        self.llm_call_fn = llm_call_fn

    @abstractmethod
    def generate(self, context: Dict) -> str:
        """
        Generate a natural language explanation from a context dictionary.

        Args:
            context: dictionary containing everything the generator needs
                     (features, values, prediction label, directions, etc.)

        Returns:
            A textual explanation produced either by templates or by an LLM.
        """
        raise NotImplementedError

    def build_prompt(self, context: Dict) -> str:
        """
        Simple fallback prompt builder.

        Subclasses can reuse this method or override it with a more
        sophisticated prompt tailored to their technique (Few-Shot, CoT, etc.).
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
        Helper: call the underlying LLM via the injected function.

        If `config.debug_print_prompt` is True, the prompt is printed to stdout
        before the call. This is very useful for debugging and for demos
        where we want to show exactly what the LLM receives.
        """
        if self.llm_call_fn is None:
            raise RuntimeError(
                "No LLM call function provided to BaseNLGGenerator. "
                "Pass llm_call_fn=... when constructing the generator."
            )

        # Optional debug: print the exact prompt sent to the LLM
        if getattr(self.config, "debug_print_prompt", False):
            print("\n================ LLM INPUT PROMPT ================")
            print(prompt)
            print("================ END OF PROMPT ==================\n")

        return self.llm_call_fn(prompt, self.config)
