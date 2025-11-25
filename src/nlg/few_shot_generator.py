# src/nlg/few_shot_generator.py
from typing import Dict, List, Optional, Callable
from .base_generator import BaseNLGGenerator


class FewShotGenerator(BaseNLGGenerator):
    """
    Few-Shot prompting based NLG generator.

    - Input: English (prediction, features, directions, method).
    - Output: English explanations.
    - audience:
        * "expert"  -> technical explanation
        * "layman"  -> simple explanation for non-expert
        * "both"    -> one answer containing both views
    """

    def __init__(
        self,
        config,
        examples: Optional[Dict[str, List[Dict]]] = None,
        llm_call_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: NLGConfig
            examples: dict with lists of examples per style:
                      {"expert": [...], "layman": [...]}
            llm_call_fn: function that calls the LLM (e.g. ollama_llm_call(prompt, config))
        """
        super().__init__(config, llm_call_fn=llm_call_fn)
        self.examples = examples or self._get_default_examples()

    # ------------------------------------------------------------------ #
    #  DEFAULT FEW-SHOT EXAMPLES (INPUT + OUTPUT IN ENGLISH)             #
    # ------------------------------------------------------------------ #

    def _get_default_examples(self) -> Dict[str, List[Dict]]:
        """
        Default few-shot examples for two audiences:
        - 'expert': technical language for clinicians / data scientists
        - 'layman': simple language for patients / non-experts
        """

        # -------- EXPERT EXAMPLES --------
        expert_examples = [
            {
                "input": (
                    "Prediction: malignant tumor\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst_area = 1200.0 (supports)\n"
                    "- mean_concave_points = 0.20 (supports)\n"
                    "- worst_radius = 18.0 (supports)\n"
                ),
                "output": (
                    "The model predicts a malignant tumor because several morphological features have large "
                    "positive contributions to the malignancy score. A very large worst area and radius are "
                    "typical of aggressive lesions and strongly support the malignant class, as reflected by "
                    "their positive SHAP values. In addition, a higher density of concave points further "
                    "reinforces this pattern. Taken together, these dominant positive contributions outweigh "
                    "any smaller opposing effects and push the overall prediction clearly towards malignancy."
                ),
            },
            {
                "input": (
                    "Prediction: benign tumor\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst_area = 420.0 (contradicts)\n"
                    "- mean_radius = 8.9 (contradicts)\n"
                    "- smoothness_mean = 0.08 (supports)\n"
                ),
                "output": (
                    "The model predicts a benign tumor because the largest area and average radius of the lesion "
                    "are within ranges typically seen in non-aggressive nodules and have negative contributions "
                    "to the malignancy score. These factors contradict a malignant pattern and dominate the overall "
                    "attribution. Although smoothness_mean has a small positive contribution, its effect is modest "
                    "compared with the stronger negative contributions. As a result, the combined evidence favours "
                    "the benign class."
                ),
            },
        ]

        # -------- LAYMAN EXAMPLES --------
        layman_examples = [
            {
                "input": (
                    "Prediction: high risk of breast cancer\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst_area = 1200.0 (supports)\n"
                    "- mean_concave_points = 0.20 (supports)\n"
                    "- worst_radius = 18.0 (supports)\n"
                ),
                "output": (
                    "The model suggests a higher risk of breast cancer because the lump is quite large and its shape "
                    "is more irregular than what we usually see in harmless findings. These characteristics are similar "
                    "to those observed in many confirmed cancer cases, so they strongly push the model towards the "
                    "cancer category. In simple terms, size and shape together make this lump look more suspicious."
                ),
            },
            {
                "input": (
                    "Prediction: low risk of breast cancer\n"
                    "Explanation method: SHAP\n"
                    "Top factors:\n"
                    "- worst_area = 420.0 (contradicts)\n"
                    "- mean_radius = 8.9 (contradicts)\n"
                    "- smoothness_mean = 0.08 (supports)\n"
                ),
                "output": (
                    "The model suggests a low risk of breast cancer because the lump is relatively small and does not "
                    "show the more aggressive patterns seen in many cancer cases. The size and overall shape of the lump "
                    "point away from cancer, even if one of the texture measurements slightly increases the risk. Overall, "
                    "the evidence leans towards the lump being harmless."
                ),
            },
        ]

        return {
            "expert": expert_examples,
            "layman": layman_examples,
        }

    # ------------------------------------------------------------------ #
    #  CONTEXT FORMATTING (INPUT 100% ENGLISH)                           #
    # ------------------------------------------------------------------ #

    def _format_context(self, context: Dict) -> str:
        """
        Format context for the prompt.

        context:
        - prediction: English string (e.g. 'malignant tumor', 'low risk of breast cancer')
        - features: feature names
        - values: contribution values (SHAP/LIME)
        - directions: ['supports'/'contradicts'/'neutral']
        - method: 'shap' / 'lime'
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

    # ------------------------------------------------------------------ #
    #  GUIDELINES (OUTPUT MUST BE ENGLISH, HIGH-FIDELITY)                #
    # ------------------------------------------------------------------ #

    def _build_guidelines(self, style: str) -> str:
        """
        Textual guidelines for the LLM.

        - Output must be in ENGLISH.
        - Do NOT invent new numbers or features.
        - Respect the directions 'supports' / 'contradicts' / 'neutral'.
        """
        style = style.lower()

        common_rules = (
            "General rules:\n"
            "- The FINAL EXPLANATION MUST BE WRITTEN ONLY IN ENGLISH.\n"
            "- Do NOT invent exact numeric percentages, thresholds or counts (like '4%', 'three times higher') "
            "that are not explicitly present in the input. Use only qualitative terms such as 'larger', 'smaller', "
            "'high', 'low' when describing magnitude, unless the number is given.\n"
            "- Do NOT introduce new features or variables that are not listed in the input.\n"
            "- Stay faithful to the provided factors and their directions (supports / contradicts / neutral).\n"
        )

        if style == "layman":
            return (
                "You are generating an explanation for a NON-EXPERT patient.\n"
                + common_rules +
                "Additional guidelines for layman explanations:\n"
                "- Use simple, calm English that a person without medical or technical background can understand.\n"
                "- Avoid technical terms like 'SHAP value', 'attribution', 'model score', 'positive class', "
                "or 'negative class'.\n"
                "- Do NOT invent mathematical operations or relationships (for example, 'ratio of means') that are "
                "not explicitly mentioned in the input.\n"
                "- Talk about how the size, shape, or structure of the lump tends to increase or decrease the risk.\n"
                "- When a factor is marked as 'supports', say that it increases or supports the predicted outcome. "
                "When it is 'contradicts', say that it goes against that outcome or lowers the risk.\n"
                "- Do not promise certainty: use expressions like 'suggests', 'indicates', 'points towards', "
                "rather than 'guarantees'.\n"
                "- Keep the explanation between 3 and 6 sentences, and keep sentences reasonably short.\n"
            )

        # expert
        return (
            "You are generating an explanation for a CLINICIAN or DATA SCIENTIST.\n"
            + common_rules +
            "Additional guidelines for expert explanations:\n"
            "- It is acceptable to refer to 'features', 'contributions', and 'model output' using technical language.\n"
            "- Do NOT change the direction provided: 'supports' must be described as supporting the predicted class, "
            "and 'contradicts' as opposing or protective with respect to that class.\n"
            "- Do NOT invent mathematical relationships that are not present in the input (for example, new ratios or "
            "combined indices).\n"
            "- Explicitly connect the sign and approximate magnitude of contributions with the final conclusion "
            "(for example, 'large positive contribution increases the malignancy score').\n"
            "- Highlight which factors are dominant and whether some factors partly counterbalance the decision.\n"
            "- Keep the explanation between 3 and 6 sentences, structured logically from evidence to conclusion.\n"
        )

    def _get_examples_for_style(self, style: str) -> List[Dict]:
        style = style.lower()
        if style in self.examples:
            return self.examples[style]
        return self.examples.get("expert", [])

    # ------------------------------------------------------------------ #
    #  PROMPT: SINGLE STYLE (expert / layman)                            #
    # ------------------------------------------------------------------ #

    def build_few_shot_prompt(self, context: Dict, style: str = "expert") -> str:
        """
        Build Few-Shot prompt for a single audience (expert OR layman).
        """
        style = style.lower()
        guidelines = self._build_guidelines(style)
        examples = self._get_examples_for_style(style)

        prompt = (
            "You are an explainable AI assistant.\n"
            "You receive model predictions and feature attributions (for example, SHAP or LIME values).\n"
            "Your task is to generate a faithful natural language explanation based ONLY on the provided factors.\n\n"
            + guidelines +
            "\nBelow are a few EXAMPLES of good explanations (Input and explanation both in English):\n\n"
        )

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input:\n{example['input']}\n"
            prompt += f"Explanation (English):\n{example['output']}\n\n"

        prompt += "--- New instance ---\n"
        prompt += self._format_context(context)
        prompt += (
            "\n\nNow write the FINAL EXPLANATION in ENGLISH only. "
            "Do not invent new features or exact numerical values.\n"
        )
        return prompt

    # ------------------------------------------------------------------ #
    #  PROMPT: BOTH VIEWS IN A SINGLE OUTPUT (EXPERT + LAYMAN)           #
    # ------------------------------------------------------------------ #

    def build_dual_prompt(self, context: Dict) -> str:
        """
        Build prompt to generate, in a single answer:

        1) A technical explanation for doctor / researcher.
        2) A simple explanation for the patient.

        Desired format of the output:

        Technical explanation (for doctor or researcher):
        <3–6 sentences in English, technical but clear>

        Simple explanation for the patient:
        <3–6 sentences in English, simple language>
        """
        expert_guidelines = self._build_guidelines("expert")
        layman_guidelines = self._build_guidelines("layman")
        expert_examples = self._get_examples_for_style("expert")
        layman_examples = self._get_examples_for_style("layman")

        prompt = (
            "You are an explainable AI assistant.\n"
            "You receive model predictions and feature attributions (for example, SHAP or LIME values).\n"
            "Your task is to generate TWO faithful natural language explanations based ONLY on the provided factors.\n"
            "Both explanations MUST be written ONLY in English.\n\n"
            "You will produce:\n"
            "1) A technical explanation for a CLINICIAN or DATA SCIENTIST.\n"
            "2) A simple explanation for a NON-EXPERT PATIENT.\n\n"
            "=== Expert guidelines ===\n"
            f"{expert_guidelines}\n"
            "=== Layman guidelines ===\n"
            f"{layman_guidelines}\n"
            "Remember: do not invent new numerical values (percentages, exact counts) or new features.\n\n"
            "Below are examples of GOOD explanations in English:\n\n"
        )

        # One expert example
        if expert_examples:
            ex = expert_examples[0]
            prompt += (
                "Expert example:\n"
                f"Input:\n{ex['input']}\n"
                f"Explanation (English):\n{ex['output']}\n\n"
            )

        # One layman example
        if layman_examples:
            ex = layman_examples[0]
            prompt += (
                "Layman example:\n"
                f"Input:\n{ex['input']}\n"
                f"Explanation (English):\n{ex['output']}\n\n"
            )

        # Context of the new instance
        prompt += "--- New instance ---\n"
        prompt += self._format_context(context)

        # Strict output format
        prompt += (
            "\n\nNow write the FINAL EXPLANATIONS in ENGLISH only.\n"
            "Use EXACTLY the following output format (do not add anything before or after):\n\n"
            "Technical explanation (for doctor or researcher):\n"
            "<3–6 sentences in English, technical explanation>\n\n"
            "Simple explanation for the patient:\n"
            "<3–6 sentences in English, simple non-technical explanation>\n"
        )

        return prompt

    # ------------------------------------------------------------------ #
    #  MOCK FALLBACK (NO LLM)                                           #
    # ------------------------------------------------------------------ #

    def _mock_generate(self, context: Dict, style: str = "expert") -> str:
        """
        Fallback if no LLM is configured.
        """
        prediction = context.get("prediction", "")
        features = context.get("features", [])
        values = context.get("values", [])
        directions = context.get("directions", [])
        parts = []
        for i, (f, v) in enumerate(zip(features, values)):
            d = directions[i] if i < len(directions) else "neutral"
            parts.append(f"{f} ({v:.2f}, {d})")

        factors = ", ".join(parts)

        if style == "layman":
            return (
                f"The model suggests '{prediction}' mainly because of the following factors: {factors}. "
                "Factors marked as 'supports' increase the likelihood of this outcome, while factors marked as "
                "'contradicts' reduce it. Overall, the balance of these factors makes this outcome the most likely."
            )
        elif style == "both":
            expert_text = (
                f"The model predicts '{prediction}' because the main factors ({factors}) have large absolute "
                "contributions to the final score. Positive contributions labelled as 'supports' push the score "
                "towards this class, whereas 'contradicts' only partially compensate them."
            )
            layman_text = (
                f"The model suggests '{prediction}' because some measurements strongly point in this direction, "
                "while others only slightly pull the result the other way. Taken together, the evidence points "
                "to this being the most likely outcome."
            )
            return (
                "Technical explanation (for doctor or researcher):\n"
                + expert_text
                + "\n\nSimple explanation for the patient:\n"
                + layman_text
            )
        else:
            return (
                f"The model predicts '{prediction}' because the main factors ({factors}) have large absolute "
                "contributions to the final score. Positive contributions labelled as 'supports' push the "
                "prediction towards this class, while 'contradicts' partially offset them without changing "
                "the final class."
            )

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                        #
    # ------------------------------------------------------------------ #

    def generate(self, context: Dict) -> str:
        """
        Generate explanation using Few-Shot prompting.

        context:
          - prediction: English string
          - features, values
          - method
          - directions: ['supports'/'contradicts'/'neutral']
          - audience:
              'expert'  -> only technical explanation
              'layman'  -> only simple explanation
              'both'    -> one answer with both views
        """
        style = context.get("audience", "expert")

        if style == "both":
            prompt = self.build_dual_prompt(context)
        else:
            prompt = self.build_few_shot_prompt(context, style=style)

        if self.llm_call_fn is not None:
            return self._call_llm(prompt)

        # Fallback mock explanation
        return self._mock_generate(context, style=style)
