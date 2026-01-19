"""
XAI-NLG Framework - Evaluation Script
Evaluează automat toate combinațiile de metode XAI și tehnici NLG.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FrameworkConfig, ExplainerConfig, NLGConfig, ValidatorConfig, NormalizerConfig
from src.pipeline import XAINLGPipeline
from src.nlg.ollama_client import ollama_llm_call
from evaluator import Evaluator, EvaluationConfig

# Setup logging
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
    """Load and prepare Breast Cancer Wisconsin dataset."""
    logger.info("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'name': 'breast_cancer'
    }


def train_models(X_train, y_train):
    """Train ML models for evaluation."""
    models = {}
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    logger.info(f"Random Forest trained - Train Acc: {rf_model.score(X_train, y_train):.3f}")
    models['random_forest'] = rf_model
    
    # Gradient Boosting (alternativă la XGBoost)
    logger.info("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    logger.info(f"Gradient Boosting trained - Train Acc: {gb_model.score(X_train, y_train):.3f}")
    models['gradient_boosting'] = gb_model
    
    return models


def setup_framework_config():
    """Setup framework configuration."""
    config = FrameworkConfig(
        explainer=ExplainerConfig(
            shap_n_samples=100,
            lime_n_samples=1000,
            top_k_features=5
        ),
        normalizer=NormalizerConfig(
            scale_method='minmax',
            feature_grouping_threshold=0.1
        ),
        nlg=NLGConfig(
            model_name='llama3:latest',
            temperature=0.3,
            max_tokens=300,
            techniques=['few_shot', 'cot', 'self_consistency'],
            debug_print_prompt=False
        ),
        validator=ValidatorConfig(
            verify_sum_conservation=True,
            sum_tolerance=0.1,
            track_evidence=True,
            min_clarity_score=40.0
        ),
        random_seed=42,
        verbose=False  # Reduce output noise during evaluation
    )
    return config


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation of all XAI methods and NLG techniques.
    """
    logger.info("=" * 70)
    logger.info("XAI-NLG FRAMEWORK - COMPREHENSIVE EVALUATION")
    logger.info("=" * 70)
    
    # Create output directory
    Path('./evaluation_results').mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    data = load_breast_cancer_data()
    
    # 2. Train models
    models = train_models(data['X_train'], data['y_train'])
    
    # 3. Setup framework
    framework_config = setup_framework_config()
    
    # 4. Define evaluation parameters
    xai_methods = ['shap', 'lime']
    nlg_techniques = ['few_shot', 'cot', 'self_consistency']
    num_instances = 5  # Number of instances to evaluate per combination
    
    # 5. Collect all results
    all_results = []
    
    # 6. Evaluate all combinations
    for model_name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*50}")
        
        for xai_method in xai_methods:
            for nlg_technique in nlg_techniques:
                logger.info(f"\n--- {xai_method.upper()} + {nlg_technique} ---")
                
                # Create pipeline with LLM
                pipeline = XAINLGPipeline(
                    model=model,
                    data=data['X_train'],
                    feature_names=data['feature_names'],
                    config=framework_config,
                    llm_call_fn=ollama_llm_call
                )
                
                # Evaluate instances
                for i in range(num_instances):
                    instance_idx = i
                    instance = data['X_test'][instance_idx]
                    true_label = data['y_test'][instance_idx]
                    
                    try:
                        # Get explanation
                        result = pipeline.explain_instance(
                            instance=instance,
                            method=xai_method,
                            technique=nlg_technique,
                            generate_text=True
                        )
                        
                        # Extract metrics
                        validation = result.get('validation', {})
                        clarity = validation.get('clarity', {}).get('score', 0)
                        coverage = validation.get('coverage', {}).get('coverage_score', 0) * 100
                        is_valid = validation.get('valid', False)
                        
                        # Get prediction info
                        prediction = result.get('prediction', -1)
                        prediction_correct = (prediction == true_label)
                        
                        # Store result
                        eval_result = {
                            'model': model_name,
                            'xai_method': xai_method,
                            'nlg_technique': nlg_technique,
                            'instance_id': f"{model_name}_{xai_method}_{nlg_technique}_{i}",
                            'true_label': true_label,
                            'prediction': prediction,
                            'prediction_correct': prediction_correct,
                            'clarity_score': clarity,
                            'coverage_score': coverage,
                            'is_valid': is_valid,
                            'text_length': len(result.get('generated_text', '')),
                            'generated_text': result.get('generated_text', '')[:200]  # First 200 chars
                        }
                        
                        all_results.append(eval_result)
                        
                        logger.info(f"  Instance {i}: Clarity={clarity:.1f}, Coverage={coverage:.0f}%, Valid={is_valid}")
                        
                    except Exception as e:
                        logger.error(f"  Error on instance {i}: {str(e)}")
                        all_results.append({
                            'model': model_name,
                            'xai_method': xai_method,
                            'nlg_technique': nlg_technique,
                            'instance_id': f"{model_name}_{xai_method}_{nlg_technique}_{i}",
                            'error': str(e)
                        })
    
    # 7. Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # 8. Export detailed results
    csv_path = './evaluation_results/detailed_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\nDetailed results saved to: {csv_path}")
    
    # 9. Generate summary statistics
    summary = generate_summary(df)
    
    # 10. Export summary
    summary_path = './evaluation_results/summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Summary report saved to: {summary_path}")
    
    # 11. Print summary
    print("\n" + summary)
    
    return df, summary


def generate_summary(df: pd.DataFrame) -> str:
    """Generate summary report from results DataFrame."""
    
    # Filter out error rows
    valid_df = df[~df['clarity_score'].isna()].copy()
    
    report = """
================================================================================
                    XAI-NLG FRAMEWORK - EVALUATION SUMMARY
================================================================================

OVERVIEW
--------
Total evaluations: {total}
Successful evaluations: {successful}
Failed evaluations: {failed}

""".format(
        total=len(df),
        successful=len(valid_df),
        failed=len(df) - len(valid_df)
    )
    
    # Overall metrics
    report += """
OVERALL METRICS
---------------
Clarity Score:  Mean={clarity_mean:.1f}, Std={clarity_std:.1f}, Min={clarity_min:.1f}, Max={clarity_max:.1f}
Coverage Score: Mean={coverage_mean:.1f}%, Std={coverage_std:.1f}%
Valid Rate:     {valid_rate:.1f}%

""".format(
        clarity_mean=valid_df['clarity_score'].mean(),
        clarity_std=valid_df['clarity_score'].std(),
        clarity_min=valid_df['clarity_score'].min(),
        clarity_max=valid_df['clarity_score'].max(),
        coverage_mean=valid_df['coverage_score'].mean(),
        coverage_std=valid_df['coverage_score'].std(),
        valid_rate=valid_df['is_valid'].mean() * 100
    )
    
    # By XAI method
    report += """
BY XAI METHOD
-------------
"""
    for method in valid_df['xai_method'].unique():
        method_df = valid_df[valid_df['xai_method'] == method]
        report += "{method:>10}: Clarity={clarity:.1f}, Coverage={coverage:.1f}%, Valid={valid:.0f}%\n".format(
            method=method.upper(),
            clarity=method_df['clarity_score'].mean(),
            coverage=method_df['coverage_score'].mean(),
            valid=method_df['is_valid'].mean() * 100
        )
    
    # By NLG technique
    report += """

BY NLG TECHNIQUE
----------------
"""
    for technique in valid_df['nlg_technique'].unique():
        tech_df = valid_df[valid_df['nlg_technique'] == technique]
        report += "{technique:>18}: Clarity={clarity:.1f}, Coverage={coverage:.1f}%, Valid={valid:.0f}%\n".format(
            technique=technique,
            clarity=tech_df['clarity_score'].mean(),
            coverage=tech_df['coverage_score'].mean(),
            valid=tech_df['is_valid'].mean() * 100
        )
    
    # By model
    report += """

BY MODEL
--------
"""
    for model in valid_df['model'].unique():
        model_df = valid_df[valid_df['model'] == model]
        report += "{model:>20}: Clarity={clarity:.1f}, Coverage={coverage:.1f}%, Valid={valid:.0f}%\n".format(
            model=model,
            clarity=model_df['clarity_score'].mean(),
            coverage=model_df['coverage_score'].mean(),
            valid=model_df['is_valid'].mean() * 100
        )
    
    # Best combinations
    report += """

BEST COMBINATIONS (by Clarity)
------------------------------
"""
    combo_stats = valid_df.groupby(['xai_method', 'nlg_technique']).agg({
        'clarity_score': 'mean',
        'coverage_score': 'mean',
        'is_valid': 'mean'
    }).round(2)
    combo_stats = combo_stats.sort_values('clarity_score', ascending=False)
    
    for idx, row in combo_stats.head(5).iterrows():
        report += f"{idx[0].upper():>6} + {idx[1]:<18}: Clarity={row['clarity_score']:.1f}, Coverage={row['coverage_score']:.1f}%\n"
    
    report += """

================================================================================
                              END OF REPORT
================================================================================
"""
    
    return report


if __name__ == '__main__':
    try:
        df, summary = run_comprehensive_evaluation()
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)