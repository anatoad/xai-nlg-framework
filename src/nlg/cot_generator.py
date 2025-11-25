# src/nlg/cot_generator.py
"""
Chain-of-Thought prompting generator.
"""

from typing import Dict, Optional, Callable, List
from .base_generator import BaseNLGGenerator


class ChainOfThoughtGenerator(BaseNLGGenerator):
    """
    Chain-of-Thought prompting based NLG generator.

    - Uses the same context format as FewShotGenerator:
      context:
        - prediction: English string (e.g. 'benign tumor', 'high risk of breast cancer')
        - features: list of feature names
        - values: list of contribution values
        - directions: list of 'supports' / 'contradicts' / 'neutral'
        - method: 'shap' / 'lime' (optional)
    """

    def __init__(
        self,
        config,
        llm_call_fn: Optional[Callable] = None,
    ):
        super().__init__(config, llm_call_fn=llm_call_fn)

    # ---------------------- helpers ---------------------- #

    def _format_context(self, context: Dict) -> str:
        """
        Same style as in FewShot: everything in English.
        """
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions")
        prediction = context.get("prediction", "")
        method = context.get("method", "")

        formatted = f"Prediction: {prediction}\n"
        if method:
            formatted += f"Explanation method: {method}\n"

        formatted += "Top factors:\n"

        if directions is not None and len(directions) == len(features):
            for f, v, d in zip(features, values, directions):
                formatted += f"- {f} = {v:.3f} ({d})\n"
        else:
            for f, v in zip(features, values):
                formatted += f"- {f} = {v:.3f}\n"

        return formatted

    def build_cot_prompt(self, context: Dict) -> str:
        """
        Build a Chain-of-Thought style prompt:
        - Ask the model to reason step by step
        - But only show the final explanation explicitly.
        """
        guidelines = (
            "You are an explainable AI assistant.\n"
            "You receive model predictions and feature attributions (for example, SHAP or LIME values).\n"
            "Your task is to reason step by step and then provide a clear explanation of the prediction.\n\n"
            "Rules:\n"
            "- The final explanation must be written ONLY in English.\n"
            "- Do NOT invent exact numeric percentages or counts that are not in the input.\n"
            "- Do NOT introduce new features that are not listed in the input.\n"
            "- Respect the directions 'supports' / 'contradicts' / 'neutral'.\n"
            "- Use the step-by-step reasoning only as an internal tool; the final explanation should be a clean paragraph.\n"
        )

        prompt = (
            guidelines
            + "\n--- Input context ---\n"
            + self._format_context(context)
            + "\n\n--- Reasoning (you can think step by step here) ---\n"
            "Think step by step about:\n"
            "- What is the predicted outcome?\n"
            "- Which factors have the largest absolute contributions?\n"
            "- Which factors support the prediction and which ones oppose it?\n"
            "- How do these factors jointly justify the final prediction?\n\n"
            "Do NOT show this internal thought process directly in the final answer.\n"
            "After you finish reasoning, write a concise explanation for the user.\n\n"
            "--- Final answer ---\n"
            "Now provide ONLY the final explanation in English (3–6 sentences):\n"
        )

        return prompt

    def _mock_generate(self, context: Dict) -> str:
        """
        Fallback if no LLM is configured.
        """
        prediction = context.get("prediction", "")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        parts: List[str] = []
        for i, (f, v) in enumerate(zip(features, values)):
            d = directions[i] if i < len(directions) else "neutral"
            parts.append(f"{f} ({v:.2f}, {d})")

        factors = ", ".join(parts)

        return (
            f"The model predicts '{prediction}' based on several key factors: {factors}. "
            "Features marked as 'supports' push the prediction towards this outcome, while those marked "
            "as 'contradicts' partially offset it. Overall, the positive contributions dominate, which explains "
            "why the model selects this outcome."
        )

    # ---------------------- public API ---------------------- #

    def generate(self, context: Dict) -> str:
        """
        Generate explanation using Chain-of-Thought prompting.
        """
        prompt = self.build_cot_prompt(context)

        if self.llm_call_fn is not None:
            return self._call_llm(prompt)

        return self._mock_generate(context)
