from typing import Dict, List
from .base_generator import BaseNLGGenerator

class FewShotGenerator(BaseNLGGenerator):
    """Few-Shot prompting based NLG generator."""
    
    def __init__(self, config, examples: List[Dict] = None):
        """
        Initialize Few-Shot generator.
        
        Args:
            config: NLG configuration
            examples: List of example explanation pairs
        """
        super().__init__(config)
        self.examples = examples or self._get_default_examples()
    
    def _get_default_examples(self) -> List[Dict]:
        """Provide default few-shot examples."""
        return [
            {
                "input": "Features: age=45, income=75000, credit_score=750",
                "output": "The applicant's high credit score (750) and stable income ($75,000) are strong positive indicators for approval."
            },
            {
                "input": "Features: age=25, income=20000, credit_score=550",
                "output": "The applicant's lower credit score (550) and modest income ($20,000) are concerning factors that may require additional review."
            }
        ]
    
    def build_few_shot_prompt(self, context: Dict) -> str:
        """Build Few-Shot prompt with examples."""
        prompt = "You are an explainable AI system. Explain model predictions clearly and concisely.\n\n"
        prompt += "Examples:\n"
        
        for i, example in enumerate(self.examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n"
        
        prompt += "\n--- New Instance ---\n"
        prompt += self._format_context(context)
        prompt += "\nExplanation:"
        
        return prompt
    
    def _format_context(self, context: Dict) -> str:
        """Format context for prompt."""
        features = context.get("features", [])
        values = context.get("values", [])
        
        formatted = "Features: "
        formatted += ", ".join(
            f"{f}={v:.3f}" for f, v in zip(features, values)
        )
        return formatted
    
    def generate(self, context: Dict) -> str:
        """Generate explanation using Few-Shot prompting."""
        prompt = self.build_few_shot_prompt(context)
        # TODO: call to llm API with prompt
        return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for demonstration."""
        return "The model's prediction is primarily driven by the top contributing factors mentioned above, which show strong alignment with the target outcome."