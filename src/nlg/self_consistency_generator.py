from typing import Dict, List
from collections import Counter
from .base_generator import BaseNLGGenerator

class SelfConsistencyGenerator(BaseNLGGenerator):
    """Self-Consistency prompting based NLG generator."""
    
    def __init__(self, config, n_chains: int = 3):
        """
        Initialize Self-Consistency generator.
        
        Args:
            config: NLG configuration
            n_chains: Number of independent reasoning chains
        """
        super().__init__(config)
        self.n_chains = n_chains
    
    def build_chain_prompt(self, context: Dict, chain_id: int) -> str:
        """Build prompt for individual chain."""
        prompt = f"""You are an explainable AI system (reasoning path {chain_id}).
Explain this model prediction independently:

{self._format_context(context)}

Provide a clear, concise explanation:"""
        return prompt
    
    def _format_context(self, context: Dict) -> str:
        """Format context for prompt."""
        features = context.get("features", [])
        values = context.get("values", [])
        prediction = context.get("prediction", "")
        
        formatted = f"Prediction: {prediction}\n"
        formatted += "Contributing factors:\n"
        for f, v in zip(features, values):
            direction = "supports" if v > 0 else "contradicts"
            formatted += f"- {f} = {v:.3f} ({direction})\n"
        return formatted
    
    def generate_single_chain(self, context: Dict) -> str:
        """Generate single reasoning chain."""
        # In real implementation, would call LLM multiple times
        return "The prediction is driven by the identified factors in their order of importance."
    
    def generate(self, context: Dict) -> str:
        """
        Generate explanation using Self-Consistency.
        
        Generates multiple chains and aggregates results.
        """
        chains = [
            self.generate_single_chain(context)
            for _ in range(self.n_chains)
        ]
        
        # Aggregate chains (majority voting on key points)
        aggregated = self._aggregate_chains(chains)
        return aggregated
    
    def _aggregate_chains(self, chains: List[str]) -> str:
        """Aggregate multiple reasoning chains."""
        if not chains:
            return ""
        
        # Simple aggregation: return most consistent chain or average
        # In real implementation, would do more sophisticated aggregation
        return chains[0]  # Placeholder
