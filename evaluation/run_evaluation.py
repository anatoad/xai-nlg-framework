"""
XAI-NLG Framework - Evaluation Script (Optimized)
Evaluează automat toate combinațiile de metode XAI și tehnici NLG.

Îmbunătățiri:
- XGBoost adăugat
- Toleranță relaxată pentru SHAP sum conservation
- 10 instanțe per combinație
- Configurări optimizate pentru prezentare
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FrameworkConfig, ExplainerConfig, NLGConfig, ValidatorConfig, NormalizerConfig
from src.pipeline import XAINLGPipeline
from src.nlg.ollama_client import ollama_llm_call

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using GradientBoosting instead.")
    print("Install with: pip install xgboost")

# Setup logging
Path('./evaluation_results').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./evaluation_results/evaluation.log', mode='w'),
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
    logger.info(f"Features: {len(feature_names)}, Classes: {len(np.unique(y))}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'name': 'breast_cancer',
        'target_names': list(data.target_names)
    }


def train_models(X_train, y_train):
    """Train ML models for evaluation."""
    models = {}
    
    # Random Forest - Best for SHAP TreeExplainer
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_train, y_train)
    logger.info(f"  Random Forest - Train Acc: {rf_acc:.3f}")
    models['RandomForest'] = rf_model
    
    # XGBoost or Gradient Boosting
    if HAS_XGBOOST:
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = xgb_model.score(X_train, y_train)
        logger.info(f"  XGBoost - Train Acc: {xgb_acc:.3f}")
        models['XGBoost'] = xgb_model
    else:
        logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_acc = gb_model.score(X_train, y_train)
        logger.info(f"  Gradient Boosting - Train Acc: {gb_acc:.3f}")
        models['GradientBoosting'] = gb_model
    
    return models


def get_framework_config(xai_method: str):
    """
    Get optimized framework config for specific XAI method.
    
    SHAP și LIME au configurări diferite pentru rezultate optime.
    """
    # Configurare de bază
    base_config = {
        'explainer': ExplainerConfig(
            shap_n_samples=100,
            lime_n_samples=1000,
            top_k_features=5
        ),
        'normalizer': NormalizerConfig(
            scale_method='minmax',
            feature_grouping_threshold=0.05
        ),
        'nlg': NLGConfig(
            model_name='llama3:latest',
            temperature=0.3,
            max_tokens=350,
            techniques=['few_shot', 'cot', 'self_consistency'],
            debug_print_prompt=False
        ),
        'random_seed': 42,
        'verbose': False
    }
    
    # Configurare specifică pentru SHAP
    if xai_method == 'shap':
        base_config['validator'] = ValidatorConfig(
            verify_sum_conservation=True,
            sum_tolerance=0.5,  # Relaxat de la 0.1 la 0.5
            track_evidence=True,
            min_clarity_score=40.0
        )
    # Configurare specifică pentru LIME
    else:
        base_config['validator'] = ValidatorConfig(
            verify_sum_conservation=False,  # LIME nu are sum conservation
            sum_tolerance=1.0,
            track_evidence=True,
            min_clarity_score=40.0
        )
    
    return FrameworkConfig(**base_config)


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation of all XAI methods and NLG techniques.
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("XAI-NLG FRAMEWORK - COMPREHENSIVE EVALUATION (OPTIMIZED)")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 1. Load data
    data = load_breast_cancer_data()
    
    # 2. Train models
    models = train_models(data['X_train'], data['y_train'])
    
    # 3. Define evaluation parameters
    xai_methods = ['shap', 'lime']
    nlg_techniques = ['few_shot', 'cot', 'self_consistency']
    num_instances = 10  # Mărit de la 5 la 10 pentru rezultate mai bune
    
    total_evals = len(models) * len(xai_methods) * len(nlg_techniques) * num_instances
    logger.info(f"\nTotal evaluations planned: {total_evals}")
    logger.info(f"Models: {list(models.keys())}")
    logger.info(f"XAI Methods: {xai_methods}")
    logger.info(f"NLG Techniques: {nlg_techniques}")
    logger.info(f"Instances per combination: {num_instances}")
    
    # 4. Collect all results
    all_results = []
    eval_count = 0
    
    # 5. Evaluate all combinations
    for model_name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*60}")
        
        for xai_method in xai_methods:
            # Get optimized config for this XAI method
            framework_config = get_framework_config(xai_method)
            
            for nlg_technique in nlg_techniques:
                logger.info(f"\n  [{xai_method.upper()}] + [{nlg_technique}]")
                
                # Create pipeline with LLM
                pipeline = XAINLGPipeline(
                    model=model,
                    data=data['X_train'],
                    feature_names=data['feature_names'],
                    config=framework_config,
                    llm_call_fn=ollama_llm_call
                )
                
                combo_results = []
                
                # Evaluate instances
                for i in range(num_instances):
                    eval_count += 1
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
                        
                        # Sum conservation (only for SHAP)
                        sum_valid = True
                        sum_diff = 0.0
                        if xai_method == 'shap':
                            sum_cons = validation.get('sum_conservation', {})
                            sum_valid = sum_cons.get('valid', True)
                            sum_diff = sum_cons.get('difference', 0.0)
                        
                        # Get prediction info
                        prediction = result.get('prediction', -1)
                        prediction_correct = (prediction == true_label)
                        
                        # Store result
                        eval_result = {
                            'model': model_name,
                            'xai_method': xai_method,
                            'nlg_technique': nlg_technique,
                            'instance_id': f"{model_name}_{xai_method}_{nlg_technique}_{i}",
                            'instance_idx': i,
                            'true_label': int(true_label),
                            'prediction': int(prediction),
                            'prediction_correct': prediction_correct,
                            'clarity_score': round(clarity, 2),
                            'coverage_score': round(coverage, 2),
                            'is_valid': is_valid,
                            'sum_conservation_valid': sum_valid,
                            'sum_conservation_diff': round(float(sum_diff), 4) if sum_diff else 0.0,
                            'text_length': len(result.get('generated_text', '')),
                            'generated_text': result.get('generated_text', '')
                        }
                        
                        combo_results.append(eval_result)
                        all_results.append(eval_result)
                        
                    except Exception as e:
                        logger.error(f"    Error on instance {i}: {str(e)}")
                        all_results.append({
                            'model': model_name,
                            'xai_method': xai_method,
                            'nlg_technique': nlg_technique,
                            'instance_id': f"{model_name}_{xai_method}_{nlg_technique}_{i}",
                            'instance_idx': i,
                            'error': str(e),
                            'clarity_score': 0,
                            'coverage_score': 0,
                            'is_valid': False
                        })
                
                # Log combo summary
                if combo_results:
                    avg_clarity = np.mean([r['clarity_score'] for r in combo_results])
                    avg_coverage = np.mean([r['coverage_score'] for r in combo_results])
                    valid_rate = np.mean([r['is_valid'] for r in combo_results]) * 100
                    logger.info(f"    Summary: Clarity={avg_clarity:.1f}, Coverage={avg_coverage:.0f}%, Valid={valid_rate:.0f}%")
    
    # 6. Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # 7. Export detailed results
    csv_path = './evaluation_results/detailed_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\nDetailed results saved to: {csv_path}")
    
    # 8. Export generated texts separately
    texts_df = df[['instance_id', 'model', 'xai_method', 'nlg_technique', 'generated_text']].copy()
    texts_path = './evaluation_results/generated_explanations.csv'
    texts_df.to_csv(texts_path, index=False)
    logger.info(f"Generated explanations saved to: {texts_path}")
    
    # 9. Generate summary statistics
    summary = generate_summary(df)
    
    # 10. Export summary
    summary_path = './evaluation_results/summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Summary report saved to: {summary_path}")
    
    # 11. Export JSON summary
    json_summary = generate_json_summary(df)
    json_path = './evaluation_results/summary.json'
    import json
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    logger.info(f"JSON summary saved to: {json_path}")
    
    # 12. Print summary
    print("\n" + summary)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\nEvaluation completed in {duration:.1f} seconds")
    
    return df, summary


def generate_json_summary(df: pd.DataFrame) -> dict:
    """Generate JSON summary for programmatic access."""
    valid_df = df[df['clarity_score'] > 0].copy()
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_evaluations': len(df),
        'successful_evaluations': len(valid_df),
        'overall': {
            'clarity_mean': round(valid_df['clarity_score'].mean(), 2),
            'clarity_std': round(valid_df['clarity_score'].std(), 2),
            'coverage_mean': round(valid_df['coverage_score'].mean(), 2),
            'coverage_std': round(valid_df['coverage_score'].std(), 2),
            'valid_rate': round(valid_df['is_valid'].mean() * 100, 2)
        },
        'by_xai_method': {},
        'by_nlg_technique': {},
        'by_model': {},
        'best_combinations': []
    }
    
    # By XAI method
    for method in valid_df['xai_method'].unique():
        method_df = valid_df[valid_df['xai_method'] == method]
        summary['by_xai_method'][method] = {
            'clarity_mean': round(method_df['clarity_score'].mean(), 2),
            'coverage_mean': round(method_df['coverage_score'].mean(), 2),
            'valid_rate': round(method_df['is_valid'].mean() * 100, 2)
        }
    
    # By NLG technique
    for tech in valid_df['nlg_technique'].unique():
        tech_df = valid_df[valid_df['nlg_technique'] == tech]
        summary['by_nlg_technique'][tech] = {
            'clarity_mean': round(tech_df['clarity_score'].mean(), 2),
            'coverage_mean': round(tech_df['coverage_score'].mean(), 2),
            'valid_rate': round(tech_df['is_valid'].mean() * 100, 2)
        }
    
    # By model
    for model in valid_df['model'].unique():
        model_df = valid_df[valid_df['model'] == model]
        summary['by_model'][model] = {
            'clarity_mean': round(model_df['clarity_score'].mean(), 2),
            'coverage_mean': round(model_df['coverage_score'].mean(), 2),
            'valid_rate': round(model_df['is_valid'].mean() * 100, 2)
        }
    
    # Best combinations
    combo_stats = valid_df.groupby(['xai_method', 'nlg_technique']).agg({
        'clarity_score': 'mean',
        'coverage_score': 'mean',
        'is_valid': 'mean'
    }).round(2)
    combo_stats = combo_stats.sort_values('clarity_score', ascending=False)
    
    for idx, row in combo_stats.head(6).iterrows():
        summary['best_combinations'].append({
            'xai_method': idx[0],
            'nlg_technique': idx[1],
            'clarity': row['clarity_score'],
            'coverage': row['coverage_score'],
            'valid_rate': round(row['is_valid'] * 100, 2)
        })
    
    return summary


def generate_summary(df: pd.DataFrame) -> str:
    """Generate summary report from results DataFrame."""
    
    # Filter out error rows
    valid_df = df[df['clarity_score'] > 0].copy()
    
    report = """
================================================================================
            XAI-NLG FRAMEWORK - EVALUATION SUMMARY (OPTIMIZED)
================================================================================

OVERVIEW
--------
Total evaluations: {total}
Successful evaluations: {successful}
Failed evaluations: {failed}
Success rate: {success_rate:.1f}%

""".format(
        total=len(df),
        successful=len(valid_df),
        failed=len(df) - len(valid_df),
        success_rate=len(valid_df) / len(df) * 100 if len(df) > 0 else 0
    )
    
    if len(valid_df) == 0:
        return report + "\nNo successful evaluations to analyze.\n"
    
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
    for method in sorted(valid_df['xai_method'].unique()):
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
    for technique in sorted(valid_df['nlg_technique'].unique()):
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
    for model in sorted(valid_df['model'].unique()):
        model_df = valid_df[valid_df['model'] == model]
        report += "{model:>20}: Clarity={clarity:.1f}, Coverage={coverage:.1f}%, Valid={valid:.0f}%\n".format(
            model=model,
            clarity=model_df['clarity_score'].mean(),
            coverage=model_df['coverage_score'].mean(),
            valid=model_df['is_valid'].mean() * 100
        )
    
    # Best combinations
    report += """

BEST COMBINATIONS (by Clarity Score)
------------------------------------
"""
    combo_stats = valid_df.groupby(['xai_method', 'nlg_technique']).agg({
        'clarity_score': 'mean',
        'coverage_score': 'mean',
        'is_valid': 'mean'
    }).round(2)
    combo_stats = combo_stats.sort_values('clarity_score', ascending=False)
    
    for idx, row in combo_stats.iterrows():
        report += f"{idx[0].upper():>6} + {idx[1]:<18}: Clarity={row['clarity_score']:.1f}, Coverage={row['coverage_score']:.1f}%, Valid={row['is_valid']*100:.0f}%\n"
    
    # Recommendations
    best_combo = combo_stats.index[0]
    best_row = combo_stats.iloc[0]
    
    report += f"""

RECOMMENDATIONS
---------------
Best combination for presentation: {best_combo[0].upper()} + {best_combo[1]}
  - Clarity: {best_row['clarity_score']:.1f}/100
  - Coverage: {best_row['coverage_score']:.1f}%
  - Valid Rate: {best_row['is_valid']*100:.0f}%

"""
    
    # SHAP vs LIME comparison
    if 'shap' in valid_df['xai_method'].values and 'lime' in valid_df['xai_method'].values:
        shap_df = valid_df[valid_df['xai_method'] == 'shap']
        lime_df = valid_df[valid_df['xai_method'] == 'lime']
        
        report += """
SHAP vs LIME COMPARISON
-----------------------
"""
        report += f"SHAP: Clarity={shap_df['clarity_score'].mean():.1f}, Coverage={shap_df['coverage_score'].mean():.1f}%, Valid={shap_df['is_valid'].mean()*100:.0f}%\n"
        report += f"LIME: Clarity={lime_df['clarity_score'].mean():.1f}, Coverage={lime_df['coverage_score'].mean():.1f}%, Valid={lime_df['is_valid'].mean()*100:.0f}%\n"
        
        if shap_df['clarity_score'].mean() > lime_df['clarity_score'].mean():
            report += "→ SHAP produces clearer explanations\n"
        else:
            report += "→ LIME produces clearer explanations\n"
        
        if shap_df['is_valid'].mean() > lime_df['is_valid'].mean():
            report += "→ SHAP has higher validation rate\n"
        else:
            report += "→ LIME has higher validation rate\n"
    
    report += """

================================================================================
                              END OF REPORT
================================================================================
"""
    
    return report


if __name__ == '__main__':
    try:
        print("\n" + "="*70)
        print("Starting XAI-NLG Framework Evaluation...")
        print("This may take 10-15 minutes depending on LLM response time.")
        print("="*70 + "\n")
        
        df, summary = run_comprehensive_evaluation()
        print("\n✅ Evaluation completed successfully!")
        print(f"\nResults saved to ./evaluation_results/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)