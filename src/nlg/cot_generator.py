"""
Chain-of-Thought NLG Generator.

Uses step-by-step reasoning to generate explanations.
"""
from typing import Dict, Optional, Callable

from .base_generator import BaseNLGGenerator


class ChainOfThoughtGenerator(BaseNLGGenerator):
    """
    Chain-of-Thought (CoT) prompting based NLG generator.
    
    Instructs the LLM to reason step-by-step before generating
    the final explanation, improving logical coherence.
    """
    
    def __init__(
        self,
        config=None,
        llm_call_fn: Optional[Callable] = None
    ):
        """
        Initialize CoT generator.
        
        Args:
            config: NLGConfig
            llm_call_fn: Function to call LLM
        """
        super().__init__(config, llm_call_fn)
    
    def _build_prompt(self, context: Dict) -> str:
        """
        Build the Chain-of-Thought prompt.
        
        Args:
            context: Context dictionary
            
        Returns:
            CoT prompt string
        """
        prompt = """You are an AI explanation assistant that reasons step-by-step.

Given model predictions and feature attributions, follow this reasoning process:

Step 1: Identify the prediction
Step 2: List the top contributing factors BY THEIR EXACT NAMES and their directions
Step 3: Analyze which factors support vs oppose the prediction
Step 4: Explain how the factors combine to produce the result
Step 5: Write a clear, concise explanation (3-5 sentences)

CRITICAL RULES:
- You MUST use the EXACT feature names from the input (e.g., "worst area", "worst concave points")
- Do NOT paraphrase or rename features (e.g., do NOT say "larger area" instead of "worst area")
- Do NOT invent numbers not in the input
- The final explanation must mention AT LEAST the top 3 features by their exact names

---
Input:
"""
        prompt += self._format_context(context)
        
        prompt += """

Now reason through this step-by-step, then provide the final explanation.
Remember: USE EXACT FEATURE NAMES from the input!

Reasoning:
Step 1: The model predicts"""
        
        return prompt
    
    def generate(self, context: Dict) -> str:
        """
        Generate explanation using Chain-of-Thought prompting.
        
        Args:
            context: Context dictionary
            
        Returns:
            Generated explanation (extracted from CoT response)
        """
        if self.llm_call_fn:
            prompt = self._build_prompt(context)
            response = self._call_llm(prompt)
            
            # Try to extract just the final explanation
            # Look for common markers
            markers = [
                "Final explanation:",
                "Final Explanation:",
                "Explanation:",
                "Step 5:",
            ]
            
            for marker in markers:
                if marker in response:
                    parts = response.split(marker)
                    if len(parts) > 1:
                        explanation = parts[-1].strip()
                        # Clean up any remaining markers
                        for m in ["Step", "Reasoning:"]:
                            if m in explanation:
                                explanation = explanation.split(m)[0].strip()
                        if len(explanation) > 50:  # Reasonable length
                            return explanation
            
            # If no marker found, try to get the last substantial paragraph
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                # Return the last paragraph that's long enough
                for p in reversed(paragraphs):
                    if len(p) > 100 and not p.startswith("Step"):
                        return p
            
            # If still nothing, return the whole response
            return response
        else:
            return self._mock_generate(context)
    
    def _mock_generate(self, context: Dict) -> str:
        """Generate mock CoT explanation."""
        prediction = context.get("prediction", "the predicted outcome")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        method = context.get("method", "XAI")
        
        # Analyze features
        supporting = [(f, v) for f, v, d in zip(features, values, directions) if d == "supports"]
        opposing = [(f, v) for f, v, d in zip(features, values, directions) if d == "contradicts"]
        
        text = f"The model predicts '{prediction}' based on {method} analysis. "
        
        if supporting:
            top_support = supporting[0]
            text += f"The primary factor driving this prediction is {top_support[0].replace('_', ' ')} "
            text += f"(contribution: {top_support[1]:.4f}), which strongly supports the outcome. "
        
        if len(supporting) > 1:
            other_support = [f[0].replace('_', ' ') for f in supporting[1:3]]
            text += f"Additional supporting factors include {', '.join(other_support)}. "
        
        if opposing:
            text += f"While {opposing[0][0].replace('_', ' ')} provides some opposing evidence, "
            text += "the supporting factors dominate the prediction. "
        
        text += "The combination of these factors leads to the final classification."
        
        return text