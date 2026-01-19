"""
Few-Shot NLG Generator.

Uses example-based prompting to generate explanations.
"""
from typing import Dict, List, Optional, Callable

from .base_generator import BaseNLGGenerator


class FewShotGenerator(BaseNLGGenerator):
    """
    Few-Shot prompting based NLG generator.
    
    Provides examples in the prompt to guide the LLM in generating
    appropriate explanations for different audiences.
    """
    
    def __init__(
        self,
        config=None,
        llm_call_fn: Optional[Callable] = None,
        examples: Optional[Dict[str, List[Dict]]] = None
    ):
        """
        Initialize Few-Shot generator.
        
        Args:
            config: NLGConfig
            llm_call_fn: Function to call LLM
            examples: Custom examples {"expert": [...], "layman": [...]}
        """
        super().__init__(config, llm_call_fn)
        self.examples = examples or self._get_default_examples()
    
    def _get_default_examples(self) -> Dict[str, List[Dict]]:
        """Get default few-shot examples for expert and layman audiences."""
        
        expert_examples = [
            {
                "input": (
                    "Prediction: malignant tumor\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst area = 0.072 (supports)\n"
                    "- worst concave points = 0.058 (supports)\n"
                    "- mean concave points = 0.041 (supports)"
                ),
                "output": (
                    "The model classifies this tumor as malignant based on several "
                    "morphological features with positive SHAP contributions. The worst area "
                    "has the highest contribution, indicating that the largest measured region "
                    "of the tumor is significantly elevated compared to typical benign cases. "
                    "Additionally, both worst and mean concave points show positive contributions, "
                    "suggesting irregular, concave boundaries characteristic of malignant growths. "
                    "These features collectively push the prediction strongly toward malignancy."
                )
            },
            {
                "input": (
                    "Prediction: benign tumor\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst radius = -0.045 (contradicts)\n"
                    "- mean perimeter = -0.038 (contradicts)\n"
                    "- texture mean = 0.012 (supports)"
                ),
                "output": (
                    "The model predicts a benign tumor primarily due to negative SHAP contributions "
                    "from size-related features. The worst radius and mean perimeter both show "
                    "negative values, indicating these measurements fall within ranges typically "
                    "associated with benign tumors rather than malignant ones. While texture mean "
                    "has a small positive contribution, its effect is minimal compared to the "
                    "dominant negative contributions from the size features, resulting in an "
                    "overall benign classification."
                )
            }
        ]
        
        layman_examples = [
            {
                "input": (
                    "Prediction: high risk\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst area = 0.072 (supports)\n"
                    "- worst concave points = 0.058 (supports)"
                ),
                "output": (
                    "The analysis suggests a higher risk because the lump appears larger than "
                    "usual and has an irregular shape. These characteristics are similar to "
                    "patterns seen in concerning cases, which is why the model flags this for "
                    "further attention. The size and shape together are the main reasons for "
                    "this assessment."
                )
            },
            {
                "input": (
                    "Prediction: low risk\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst radius = -0.045 (contradicts)\n"
                    "- mean perimeter = -0.038 (contradicts)"
                ),
                "output": (
                    "The analysis suggests a lower risk because the measurements are within "
                    "normal ranges. The size of the area of concern is typical of harmless "
                    "findings, and its overall dimensions don't show the patterns usually "
                    "associated with problems. These factors together indicate a reassuring result."
                )
            }
        ]
        
        return {
            "expert": expert_examples,
            "layman": layman_examples
        }
    
    def _build_prompt(self, context: Dict, audience: str = "expert") -> str:
        """
        Build the few-shot prompt.
        
        Args:
            context: Context dictionary with prediction, features, etc.
            audience: "expert" or "layman"
            
        Returns:
            Complete prompt string
        """
        examples = self.examples.get(audience, self.examples["expert"])
        
        # System instructions
        if audience == "layman":
            instructions = (
                "You are explaining AI predictions to a non-technical person. "
                "Use simple, clear language without jargon. "
                "Focus on what the results mean practically, not technical details."
            )
        else:
            instructions = (
                "You are explaining AI predictions to a technical audience. "
                "Use precise terminology and reference the specific feature contributions. "
                "Explain how the features combine to produce the prediction."
            )
        
        prompt = f"""You are an AI explanation assistant.

{instructions}

Rules:
- Write the explanation in clear, grammatically correct English
- Do NOT invent numbers or percentages not in the input
- Do NOT mention features not listed in the input
- Keep the explanation to 3-5 sentences
- Explain how the factors work together

Here are examples of good explanations:

"""
        
        # Add examples
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input:\n{ex['input']}\n\n"
            prompt += f"Explanation:\n{ex['output']}\n\n"
        
        # Add current instance
        prompt += "---\nNow explain this case:\n\n"
        prompt += self._format_context(context)
        prompt += "\n\nExplanation:"
        
        return prompt
    
    def generate(self, context: Dict) -> str:
        """
        Generate explanation using few-shot prompting.
        
        Args:
            context: Context dictionary with:
                - prediction, features, values, directions, method
                - audience (optional): "expert", "layman", or "both"
                
        Returns:
            Generated explanation text
        """
        audience = context.get("audience", "expert")
        
        if audience == "both":
            # Generate both expert and layman explanations
            expert_prompt = self._build_prompt(context, "expert")
            layman_prompt = self._build_prompt(context, "layman")
            
            if self.llm_call_fn:
                expert_text = self._call_llm(expert_prompt)
                layman_text = self._call_llm(layman_prompt)
            else:
                expert_text = self._mock_generate(context)
                layman_text = self._mock_generate_layman(context)
            
            return (
                f"Technical Explanation:\n{expert_text}\n\n"
                f"Simple Explanation:\n{layman_text}"
            )
        
        prompt = self._build_prompt(context, audience)
        
        if self.llm_call_fn:
            return self._call_llm(prompt)
        else:
            if audience == "layman":
                return self._mock_generate_layman(context)
            return self._mock_generate(context)
    
    def _mock_generate_layman(self, context: Dict) -> str:
        """Generate a simple mock explanation for non-technical audience."""
        prediction = context.get("prediction", "this result")
        features = context.get("features", [])
        directions = context.get("directions", [])
        
        supporting = [f.replace("_", " ") for f, d in zip(features, directions) if d == "supports"]
        opposing = [f.replace("_", " ") for f, d in zip(features, directions) if d == "contradicts"]
        
        text = f"The analysis indicates '{prediction}'. "
        
        if supporting:
            text += f"The main factors pointing to this result are: {', '.join(supporting[:2])}. "
        if opposing:
            text += f"Some factors ({', '.join(opposing[:1])}) suggest otherwise, but they are outweighed. "
        
        text += "Overall, the evidence points toward this conclusion."
        
        return text