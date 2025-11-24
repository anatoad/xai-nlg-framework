"""Chain-of-Thought prompting generator."""
from typing import Dict, List
from .base_generator import BaseNLGGenerator

class ChainOfThoughtGenerator(BaseNLGGenerator):
    """Chain-of-Thought prompting based NLG generator."""
    
    def __init__(self, config):
        """Initialize CoT generator."""
        super().__init__(config)
    
    def build_cot_prompt(self, context: Dict) -> str:
        """Build Chain-of-Thought prompt."""
        prompt = """You are an explainable AI system. Explain model predictions step-by-step.

Follow this structure:
1. Identify the prediction and target
2. List key contributing factors
3. Analyze their relationships
4. Provide concise final explanation

--- Analysis Section (Internal Reasoning) ---
Think through the following:
- What is the prediction?
- Which factors contribute most?
- What is their relative importance?
- How do they interact?

--- Final Explanation Section ---
"""
        prompt += self._format_context(context)
        prompt += "\n\nExplanation:"
        return prompt
    
    def _format_context(self, context: Dict) -> str:
        """Format context for CoT."""
        features = context.get("features", [])
        values = context.get("values", [])
        prediction = context.get("prediction", "")
        
        formatted = f"Prediction: {prediction}\n"
        formatted += "Key factors:\n"
        for f, v in zip(features, values):
            importance = "strong" if abs(v) > 0.5 else "moderate" if abs(v) > 0.2 else "weak"
            formatted += f"- {f}: {v:.3f} ({importance})\n"
        return formatted
    
    def generate(self, context: Dict) -> str:
        """Generate explanation using Chain-of-Thought."""
        prompt = self.build_cot_prompt(context)
        # In real implementation, call LLM API here
        return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for demonstration."""
        return "Based on the step-by-step analysis, the primary drivers of this prediction are the high-impact factors, which together explain the model's decision with strong confidence."
