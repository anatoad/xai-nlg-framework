"""
Base class for NLG generators.
"""
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional


class BaseNLGGenerator(ABC):
    """
    Abstract base class for all NLG generators.
    
    Generators convert structured XAI explanations into natural language text.
    """
    
    def __init__(
        self,
        config=None,
        llm_call_fn: Optional[Callable[[str, any], str]] = None
    ):
        """
        Initialize the generator.
        
        Args:
            config: NLGConfig object
            llm_call_fn: Function to call LLM: (prompt, config) -> response
        """
        self.config = config
        self.llm_call_fn = llm_call_fn
        
        # Extract config values
        if config:
            self.model_name = config.model_name
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.debug = getattr(config, 'debug_print_prompt', False)
        else:
            self.model_name = "llama3:latest"
            self.temperature = 0.3
            self.max_tokens = 300
            self.debug = False
    
    @abstractmethod
    def generate(self, context: Dict) -> str:
        """
        Generate natural language explanation from context.
        
        Args:
            context: Dictionary containing:
                - prediction: str
                - features: List[str]
                - values: List[float]
                - directions: List[str] ("supports"/"contradicts"/"neutral")
                - method: str ("SHAP"/"LIME")
                
        Returns:
            Generated explanation text
        """
        pass
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        if self.debug:
            print("\n" + "="*60)
            print("LLM PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
        
        if self.llm_call_fn is None:
            raise RuntimeError(
                "No LLM call function provided. Either pass llm_call_fn "
                "or implement _mock_generate() for testing."
            )
        
        response = self.llm_call_fn(prompt, self.config)
        return str(response).strip()
    
    def _format_context(self, context: Dict) -> str:
        """
        Format context into a string for the prompt.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted string
        """
        prediction = context.get("prediction", "unknown")
        method = context.get("method", "XAI")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        
        lines = [
            f"Prediction: {prediction}",
            f"Explanation method: {method}",
            "Top contributing factors:"
        ]
        
        for i, (feat, val) in enumerate(zip(features, values)):
            direction = directions[i] if i < len(directions) else "unknown"
            lines.append(f"  - {feat} = {val:.4f} ({direction})")
        
        return "\n".join(lines)
    
    def _mock_generate(self, context: Dict) -> str:
        """
        Generate a mock explanation without calling LLM.
        Override in subclasses for testing.
        
        Args:
            context: Context dictionary
            
        Returns:
            Mock explanation text
        """
        prediction = context.get("prediction", "the predicted outcome")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        
        # Build simple explanation
        parts = []
        for i, (feat, val) in enumerate(zip(features[:3], values[:3])):
            direction = directions[i] if i < len(directions) else "affects"
            feat_display = feat.replace("_", " ")
            if direction == "supports":
                parts.append(f"{feat_display} (value: {val:.3f}) supports this prediction")
            elif direction == "contradicts":
                parts.append(f"{feat_display} (value: {val:.3f}) opposes this prediction")
            else:
                parts.append(f"{feat_display} (value: {val:.3f}) has neutral impact")
        
        factors_text = ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0] if parts else "the input features"
        
        return (
            f"The model predicts '{prediction}' based on several key factors. "
            f"Specifically, {factors_text}. "
            f"The combination of these factors leads the model to this conclusion."
        )