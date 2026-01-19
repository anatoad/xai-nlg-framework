import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config.settings import FrameworkConfig
from src.pipeline import XAINLGPipeline
from src.nlg.ollama_client import ollama_llm_call


# load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = list(data.feature_names)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# create pipeline
config = FrameworkConfig()
pipeline = XAINLGPipeline(
    model,
    X_train,
    feature_names,
    config=config,
    llm_call_fn=ollama_llm_call  # Adăugat!
)

# explain instance
instance = X_test[0]
#result = pipeline.explain_instance(instance, method="shap", generate_text=True)
#result = pipeline.explain_instance(instance, method="lime", generate_text=True)
#result = pipeline.explain_instance(instance, method="shap", technique="cot", generate_text=True)
result = pipeline.explain_instance(instance, method="shap", technique="self_consistency", generate_text=True)

# print results
print("=== XAI-NLG Framework Results ===")
print(f"Prediction: {result['prediction']}")
print(f"Generated Explanation: {result['generated_text']}")
print(f"Validation Score: {result['validation']['clarity']['score']:.2f}")

print(result["ranked_features"][:5])  # top 5 (feature, contribution)

for fs in result["statements"][:5]:
    print(fs.rank, fs.feature, fs.value, "->", fs.statement)

print(result["validation"])  # clarity score, sum conservation etc.

# track evidence
pipeline.evidence_tracker.add_record(
    instance_id="test_instance_0",
    method=result["method"],
    prediction=str(result["prediction"]),
    contributions={fs.feature: fs.value for fs in result["statements"][:5]},
    generated_text=result["generated_text"],
    nlg_technique="few_shot",
    validation_results=result["validation"],
)

# export evidence
pipeline.evidence_tracker.export_csv("evidence_audit.csv")
