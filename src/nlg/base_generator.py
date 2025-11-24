from abc import ABC, abstractmethod
from typing import Dict, List
from config.settings import NLGConfig

class BaseNLGGenerator(ABC):
    def __init__(self, config: NLGConfig):
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    @abstractmethod
    def generate(self, context: Dict) -> str:
        """
        Generate natural language explanation.
        
        Args:
            context: Dictionary with explanation context
            
        Returns:
            Generated text
        """
        pass
    
    def build_prompt(self, context: Dict) -> str:
        """Build prompt from context."""
        features = context.get("features", [])
        values = context.get("values", [])
        prediction = context.get("prediction", "")
        
        prompt = f"Explain why the model predicted {prediction}.\n\n"
        prompt += "Key contributing factors:\n"
        for feature, value in zip(features, values):
            prompt += f"- {feature}: {value:.3f}\n"
        
        return prompt