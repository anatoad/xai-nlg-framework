import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging

sys.path.insert(0, '../')
from config.settings import FrameworkConfig, ExplainerConfig, NLGConfig, ValidatorConfig, NormalizerConfig
from src.pipeline import XAINLGPipeline
from evaluator import Evaluator, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./evaluation_results/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_breast_cancer_data():
    logger.info("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Breast Cancer dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'name': 'breast_cancer'
    }

def train_models(X_train, X_test, y_train, y_test, feature_names):
    models = {}
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    logger.info(f"RF - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    models['random_forest'] = rf_model
    
    # XGBoost
    # logger.info("Training XGBoost...")
    # xgb_model = xgb.XGBClassifier(
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     random_state=42,
    #     eval_metric='logloss'
    # )
    # xgb_model.fit(X_train, y_train)
    # train_acc = xgb_model.score(X_train, y_train)
    # test_acc = xgb_model.score(X_test, y_test)
    # logger.info(f"XGB - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    # models['xgboost'] = xgb_model
    
    return models

def setup_framework_config():
    config = FrameworkConfig(
        explainer=ExplainerConfig(
            shap_n_samples=100,
            lime_n_samples=1000,
            top_k_features=5
        ),
        normalizer=NormalizerConfig(
            scale_method='minmax',
            feature_grouping_threshold=0.1,
            text_template_style='concise'
        ),
        nlg=NLGConfig(
            model_name='llama3:latest',
            temperature=0.3,
            max_tokens=200,
            techniques=['few_shot', 'cot', 'self_consistency'],
            debug_print_prompt=False
        ),
        validator=ValidatorConfig(
            verify_sum_conservation=True,
            sum_tolerance=1e-5,
            track_evidence=True,
            min_clarity_score=45.0,
            min_robustness_jaccard=0.6
        ),
        random_seed=42,
        verbose=True
    )
    return config

def run_evaluation():
    logger.info("=" * 70)
    logger.info("XAI-NLG FRAMEWORK - EVALUATION PART III")
    logger.info("=" * 70)
    
    # 1. Load data
    breast_cancer_data = load_breast_cancer_data()
    
    # 2. Train models
    models = train_models(
        breast_cancer_data['X_train'],
        breast_cancer_data['X_test'],
        breast_cancer_data['y_train'],
        breast_cancer_data['y_test'],
        breast_cancer_data['feature_names']
    )
    
    # 3. Setup framework
    framework_config = setup_framework_config()
    
    # 4. Setup evaluator
    eval_config = EvaluationConfig(
        num_instances=20,
        num_robustness_runs=5,
        datasets=['breast_cancer'],
        methods=['shap', 'lime'],
        top_k=5,
        output_dir='./evaluation_results'
    )
    evaluator = Evaluator(eval_config)
    
    # 5. Evaluare pentru fiecare model și metodă
    logger.info("Starting evaluation...")
    
    for model_name, model in models.items():
        logger.info(f"\n--- Evaluating {model_name} ---")
        
        for method in eval_config.methods:
            logger.info(f"\n  Using method: {method}")

            pipeline = XAINLGPipeline(
                model=model,
                data=breast_cancer_data['X_train'],
                feature_names=breast_cancer_data['feature_names'],
                config=framework_config
            )

            # evaluate batch
            try:
                batch_results = evaluator.evaluate_batch(
                    X=breast_cancer_data['X_test'][:4],
                    y=breast_cancer_data['y_test'][:4],
                    X_test_ids=[str(i) for i in range(len(breast_cancer_data['X_test'][:4]))],
                    pipeline=pipeline,
                    dataset_name='breast_cancer',
                    method=method,
                    evaluate_robustness=True
                )

                # Log detailed results for debugging
                for result in batch_results:
                    logger.debug(f"Instance ID: {result.instance_id}, Clarity: {result.clarity_score}, Coverage: {result.coverage_topk}, Fidelity: {result.fidelity_score}")

                logger.info(f"  Successfully evaluated {len(batch_results)} instances")
            except Exception as e:
                logger.error(f"Error during batch evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 6. Aggregate results
    logger.info("\n" + "=" * 70)
    logger.info("Aggregating results...")
    summary = evaluator.aggregate_results()
    
    # 7. Export results
    logger.info("Exporting results...")
    csv_path = evaluator.export_results_csv()
    json_path = evaluator.export_summary_json()
    
    # 8. Generate report
    report = evaluator.generate_report()
    report_path = './evaluation_results/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    print(report)
    
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results exported to:")
    logger.info(f"  - CSV: {csv_path}")
    logger.info(f"  - JSON Summary: {json_path}")
    logger.info(f"  - Report: {report_path}")
    logger.info(f"  - Log: ./evaluation_results/evaluation.log")
    
    return evaluator, summary

if __name__ == '__main__':
    try:
        evaluator, summary = run_evaluation()
        print("\nEvaluation completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
