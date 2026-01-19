"""
Self-Consistency NLG Generator.

Generates multiple explanations and aggregates them for robustness.
"""
from typing import Dict, List, Optional, Callable
from collections import Counter

from .base_generator import BaseNLGGenerator


class SelfConsistencyGenerator(BaseNLGGenerator):
    """
    Self-Consistency prompting based NLG generator.
    
    Generates multiple independent explanations and aggregates them
    to produce a more robust final explanation.
    """
    
    def __init__(
        self,
        config=None,
        llm_call_fn: Optional[Callable] = None,
        n_samples: int = 3
    ):
        """
        Initialize Self-Consistency generator.
        
        Args:
            config: NLGConfig
            llm_call_fn: Function to call LLM
            n_samples: Number of independent explanations to generate
        """
        super().__init__(config, llm_call_fn)
        self.n_samples = n_samples
    
    def _build_single_prompt(self, context: Dict, sample_id: int) -> str:
        """
        Build prompt for a single explanation sample.
        
        Args:
            context: Context dictionary
            sample_id: Sample number (for variation)
            
        Returns:
            Prompt string
        """
        # Slightly vary the instructions for diversity
        variations = [
            "Focus on the most important factors first.",
            "Consider how the factors interact with each other.",
            "Emphasize the practical implications of each factor.",
        ]
        
        variation = variations[sample_id % len(variations)]
        
        prompt = f"""You are an AI explanation assistant (perspective {sample_id + 1}).

Generate a clear explanation for this model prediction.
{variation}

Rules:
- Write 3-5 sentences in clear English
- Only reference factors from the input
- Do not invent statistics or numbers

Input:
{self._format_context(context)}

Explanation:"""
        
        return prompt
    
    def _build_aggregation_prompt(
        self,
        context: Dict,
        explanations: List[str]
    ) -> str:
        """
        Build prompt to aggregate multiple explanations.
        
        Args:
            context: Original context
            explanations: List of generated explanations
            
        Returns:
            Aggregation prompt
        """
        prompt = """You are an AI assistant that combines multiple explanations into one.

Given several explanations for the same prediction, create a single, coherent explanation that:
- Captures the key points mentioned across explanations
- Prioritizes points that appear in multiple explanations
- Is clear and well-structured (3-5 sentences)
- Does not add any information not present in the original explanations

Original prediction context:
"""
        prompt += self._format_context(context)
        
        prompt += "\n\nExplanations to combine:\n"
        for i, exp in enumerate(explanations, 1):
            prompt += f"\nExplanation {i}:\n{exp}\n"
        
        prompt += "\nCombined explanation:"
        
        return prompt
    
    def generate(self, context: Dict) -> str:
        """
        Generate explanation using self-consistency.
        
        Args:
            context: Context dictionary
            
        Returns:
            Aggregated explanation
        """
        if self.llm_call_fn:
            # Generate multiple explanations
            explanations = []
            for i in range(self.n_samples):
                prompt = self._build_single_prompt(context, i)
                exp = self._call_llm(prompt)
                explanations.append(exp.strip())
            
            # If all explanations are identical, return one
            if len(set(explanations)) == 1:
                return explanations[0]
            
            # Aggregate using LLM
            agg_prompt = self._build_aggregation_prompt(context, explanations)
            aggregated = self._call_llm(agg_prompt)
            
            return aggregated.strip()
        else:
            return self._mock_generate(context)
    
    def _mock_generate(self, context: Dict) -> str:
        """Generate mock self-consistency explanation."""
        prediction = context.get("prediction", "the predicted outcome")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        method = context.get("method", "XAI")
        
        # Simulate aggregation by creating a comprehensive explanation
        supporting = [(f, v) for f, v, d in zip(features, values, directions) if d == "supports"]
        opposing = [(f, v) for f, v, d in zip(features, values, directions) if d == "contradicts"]
        
        text = f"Based on multiple analysis perspectives, the model predicts '{prediction}'. "
        
        if supporting:
            support_names = [f[0].replace('_', ' ') for f in supporting[:2]]
            text += f"The consistent finding across analyses is that {' and '.join(support_names)} "
            text += "are the primary factors supporting this prediction. "
        
        if opposing:
            text += f"While {opposing[0][0].replace('_', ' ')} shows some opposing contribution, "
            text += "this is consistently outweighed by the supporting factors. "
        
        text += (
            "The aggregated analysis provides high confidence in this conclusion, "
            "as the key factors are identified consistently across independent reasoning paths."
        )
        
        return text