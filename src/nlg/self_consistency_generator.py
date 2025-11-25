from typing import Dict, Optional, Callable, List
from collections import Counter
from .base_generator import BaseNLGGenerator


class SelfConsistencyGenerator(BaseNLGGenerator):
    """
    Self-Consistency prompting based NLG generator.
    - Generates multiple independent explanations (chains)
    - Aggregates them using an LLM-based summary (if available)
    - Falls back to a simple majority-vote / first-chain strategy otherwise

    - Uses the same context structure as FewShot / CoT:
      {
        "features": [...],
        "values": [...],
        "directions": [...],   # optional
        "prediction": "...",
        "method": "shap" / "lime"
      }
    """

    def __init__(
        self,
        config,
        n_chains: int = 3,
        llm_call_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config: NLGConfig
            n_chains: number of independent reasoning chains
            llm_call_fn: LLM call function (e.g. ollama_llm_call)
        """
        super().__init__(config, llm_call_fn=llm_call_fn)
        self.n_chains = max(1, int(n_chains))

    def _format_context(self, context: Dict) -> str:
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

    def build_chain_prompt(self, context: Dict, chain_id: int) -> str:
        """
        Build prompt for a single reasoning chain.
        """
        prompt = (
            f"You are an explainable AI assistant (reasoning path {chain_id}).\n"
            "You receive model predictions and feature attributions (for example, SHAP or LIME values).\n"
            "Explain this model prediction independently.\n\n"
            "Rules:\n"
            "- The explanation must be written ONLY in English.\n"
            "- Do NOT invent exact numeric percentages or counts that are not in the input.\n"
            "- Do NOT introduce new features that are not listed in the input.\n"
            "- Respect the 'supports' / 'contradicts' / 'neutral' directions.\n"
            "- Provide a concise explanation in 3–6 sentences.\n\n"
            "--- Input context ---\n"
            f"{self._format_context(context)}\n\n"
            "Now write the explanation:\n"
        )
        return prompt

    def generate_single_chain(self, context: Dict, chain_id: int) -> str:
        """
        Generate a single explanation chain.
        """
        prompt = self.build_chain_prompt(context, chain_id)

        if self.llm_call_fn is not None:
            text = self._call_llm(prompt)
            return str(text).strip()

        # fallback mock
        prediction = context.get("prediction", "")
        return (
            f"(chain {chain_id}) The model suggests '{prediction}' based on the pattern of contributions "
            "from the most important features."
        )

    def _build_aggregation_prompt(self, chains: List[str], context: Dict) -> str:
        """
        Build a prompt that shows all chains and asks the LLM
        to aggregate them into a single, consistent explanation.
        """
        prediction = context.get("prediction", "")
        base_ctx = self._format_context(context)

        prompt = (
            "You are an explainable AI assistant.\n"
            "You are given several independent explanations generated for the same model prediction.\n"
            "Your task is to aggregate them into a single, coherent explanation.\n\n"
            "Rules:\n"
            "- Write the final explanation ONLY in English.\n"
            "- Do NOT invent exact numeric percentages or counts that are not in the input.\n"
            "- Do NOT introduce new features that are not listed in the context or in the explanations.\n"
            "- Focus on points that appear in at least two explanations, or are clearly consistent.\n"
            "- Provide a clear explanation in 3–6 sentences.\n\n"
            "--- Prediction & top factors (context) ---\n"
            f"{base_ctx}\n\n"
            "--- Independent explanations ---\n"
        )

        for i, c in enumerate(chains, start=1):
            prompt += f"\nExplanation {i}:\n{c.strip()}\n"

        prompt += (
            "\n--- Task ---\n"
            "Combine the key consistent ideas from these explanations into ONE final explanation.\n"
            "Write only the final explanation, with no preamble or bullet points.\n"
        )

        return prompt

    def _aggregate_chains(self, chains: List[str], context: Dict) -> str:
        """
        Aggregate multiple reasoning chains.

        Strategy:
        1) If an LLM is available: use it to produce an aggregated summary
           over all chains (LLM-based self-consistency).
        2) If no LLM or something fails:
           - if one explanation appears more than once, return the most frequent;
           - otherwise, return the first non-empty explanation.
        """
        clean_chains = [c.strip() for c in chains if c and c.strip()]
        if not clean_chains:
            return ""

        # 1) Preferred path: LLM-based aggregation
        if self.llm_call_fn is not None:
            try:
                agg_prompt = self._build_aggregation_prompt(clean_chains, context)
                aggregated = self._call_llm(agg_prompt)
                aggregated = str(aggregated).strip()
                if aggregated:
                    return aggregated
            except Exception:
                pass

        # 2) Fallback: majority vote / first non-empty
        counts = Counter(clean_chains)
        most_common, freq = counts.most_common(1)[0]

        if freq > 1:
            return most_common

        # all are unique → just return the first
        return clean_chains[0]

    def generate(self, context: Dict) -> str:
        """
        Generate explanation using Self-Consistency:
        - sample multiple independent explanations
        - aggregate them into a single final explanation
        """
        chains: List[str] = []
        for i in range(1, self.n_chains + 1):
            chains.append(self.generate_single_chain(context, chain_id=i))

        aggregated = self._aggregate_chains(chains, context)
        if aggregated:
            return aggregated

        # fallback if something went wrong
        prediction = context.get("prediction", "")
        return (
            f"The model suggests '{prediction}' based on a consistent pattern of feature contributions "
            "across multiple independent reasoning chains."
        )
