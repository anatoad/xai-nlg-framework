import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
from datetime import datetime

sys.path.insert(0, '../')
from src.pipeline import XAINLGPipeline
from metrics import (
    FidelityMetrics, RobustnessMetrics, UsefulnessMetrics, 
    MetricsResult
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    num_instances: int = 20
    num_robustness_runs: int = 5
    datasets: List[str] = None
    methods: List[str] = None  # ['shap', 'lime']
    top_k: int = 5
    output_dir: str = './evaluation_results'
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ['breast_cancer']
        if self.methods is None:
            self.methods = ['shap']

class Evaluator:
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results = []
        self.summary = {}
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def evaluate_instance(
        self,
        pipeline: XAINLGPipeline,
        x: np.ndarray,
        y_true: int,
        instance_id: str,
        dataset_name: str = 'unknown',
        method: str = 'shap'
    ) -> MetricsResult:
        """
        Evaluate an individual instance
        """
        
        try:
            # explain instance
            explanation_result = pipeline.explain_instance(x)
            
            # extract information for evaluation
            generated_text = explanation_result.get('generated_text', '')
            top_k = pipeline.config.explainer.top_k_features
            features_topk = [str(feature_map[0]) for feature_map in explanation_result.get('ranked_features', '')[:top_k]]
            predictions = explanation_result.get('prediction', 0)
            validation_summary = explanation_result.get('validation_summary', {})
            
            # get contributions from validator
            contributions = {}
            for fs in explanation_result.get('statements', []):
                contributions[fs.feature] = fs.value
            
            # calculate fidelity metrics
            fidelity_score, sign_details = FidelityMetrics.sign_match_score(
                contributions,
                features_topk,
                generated_text,
                top_k=self.config.top_k
            )

            # SHAP conservation metrics
            shap_conservation_valid, shap_conservation_error = FidelityMetrics.shap_sum_conservation(
                explanation_result.get('explanation', {}),
                predictions,
                explanation_result.get('base_value', 0.0)
            )

            # usefulness metrics
            clarity_score = UsefulnessMetrics.clarity_score(
                generated_text,
                features_topk
            )
            
            coverage_topk = UsefulnessMetrics.coverage_topk(
                generated_text,
                features_topk,
                k=self.config.top_k
            )

            result = MetricsResult(
                instance_id=instance_id,
                fidelity_score=fidelity_score,
                sign_match_score=sign_details.get('correct_signs', 0) / max(1, sign_details.get('total_features', 1)),
                clarity_score=clarity_score,
                coverage_topk=coverage_topk,
                shap_conservation_valid=shap_conservation_valid,
                shap_conservation_error=float(shap_conservation_error) if shap_conservation_error else 0.0,
                dataset=dataset_name,
                method=method
            )
            
            logger.info(f"Instance {instance_id}: clarity={clarity_score:.1f}, coverage={coverage_topk:.1f}%, fidelity={fidelity_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating instance {instance_id}: {str(e)}")
            return MetricsResult(
                instance_id=instance_id,
                fidelity_score=0.0,
                sign_match_score=0.0,
                clarity_score=0.0,
                coverage_topk=0.0,
                shap_conservation_valid=False,
                shap_conservation_error=0.0,
                dataset=dataset_name,
                method=method
            )
    
    def evaluate_robustness(
        self,
        pipeline: XAINLGPipeline,
        x: np.ndarray,
        instance_id: str,
        dataset_name: str = 'unknown',
        method: str = 'shap',
        num_runs: int = 5
    ) -> Dict:
        """
        Evaluate robustness using multiple runs with diferent seeds
        """
        contributions_list = []
        ranks_list = []
        
        for run_idx in range(num_runs):
            try:
                # Setează seed diferit
                pipeline.config.random_seed = 42 + run_idx
                
                result = pipeline.explain_instance(x)
                
                # extract contributions and rankings
                contributions = {}
                for fs in result.get('feature_statements', []):
                    contributions[fs.feature] = fs.value
                
                contributions_list.append(contributions)
                
                # create rankings
                ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                ranks_list.append(ranked)
                
            except Exception as e:
                logger.warning(f"Robustness run {run_idx} failed: {str(e)}")
        
        if len(contributions_list) < 2:
            return {'error': 'Insufficient successful runs'}
        
        # calculate jaccard 
        features_list = [f for f, _ in ranks_list[0]]
        jaccard_scores = []
        for i in range(1, len(ranks_list)):
            j_score = RobustnessMetrics.jaccard_at_k(
                features_list,
                [f for f, _ in ranks_list[i]],
                k=self.config.top_k
            )
            jaccard_scores.append(j_score)
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        
        # calculate contribution stability
        stability_score, instability_flags = RobustnessMetrics.contribution_stability(
            contributions_list,
            top_k=self.config.top_k
        )
        
        # calculate Spearman correlation
        if len(ranks_list) >= 2:
            spearman_corr = RobustnessMetrics.ranking_consistency(
                ranks_list[0],
                ranks_list[1]
            )
        else:
            spearman_corr = 1.0
        
        return {
            'jaccard_mean': float(avg_jaccard),
            'jaccard_scores': [float(j) for j in jaccard_scores],
            'stability_score': float(stability_score),
            'spearman_correlation': float(spearman_corr),
            'instability_flags': instability_flags,
            'num_successful_runs': len(contributions_list)
        }
    
    def evaluate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test_ids: List[str],
        pipeline: XAINLGPipeline,
        dataset_name: str = 'unknown',
        method: str = 'shap',
        evaluate_robustness: bool = False
    ) -> List[MetricsResult]:
        """
        Evaluate a batch of instances
        """
        batch_results = []
        
        num_to_evaluate = min(self.config.num_instances, len(X))
        indices = np.random.choice(len(X), size=num_to_evaluate, replace=False)
        
        logger.info(f"Evaluating {num_to_evaluate} instances from {dataset_name} using {method}...")
        
        # Check if the CSV file exists
        csv_path = self.config.output_dir + '/evaluation_results.csv'
        try:
            existing_results = pd.read_csv(csv_path)
            existing_ids = set(existing_results['instance_id'])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            existing_results = pd.DataFrame()
            existing_ids = set()

        for idx, sample_idx in enumerate(indices):
            instance_id = f"{dataset_name}_{method}_{X_test_ids[sample_idx]}"

            if instance_id in existing_ids:
                logger.info(f"Skipping already evaluated instance: {instance_id}")
                continue

            # Standard instance evaluation
            result = self.evaluate_instance(
                pipeline,
                X[sample_idx],
                y[sample_idx],
                instance_id,
                dataset_name,
                method
            )

            if evaluate_robustness:
                robustness_metrics = self.evaluate_robustness(
                    pipeline,
                    X[sample_idx],
                    instance_id,
                    dataset_name,
                    method,
                    num_runs=self.config.num_robustness_runs
                )
                result.jaccard_score = robustness_metrics.get('jaccard_mean', None)
                result.ranking_stability = robustness_metrics.get('stability_score', None)
                result.spearman_correlation = robustness_metrics.get('spearman_correlation', None)

            batch_results.append(result)
            self.results.append(result)

            # Append new results to the CSV file
            result_df = pd.DataFrame([r.to_dict() for r in [result]])
            result_df.to_csv(csv_path, mode='a', header=existing_results.empty, index=False)

        # Write summary JSON after each batch
        self.aggregate_results()
        with open(self.config.output_dir + '/summary.json', 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        return batch_results
    
    def aggregate_results(self) -> Dict:
        if not self.results:
            return {}
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        summary = {
            'total_instances': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # global metrics
        for col in ['fidelity_score', 'sign_match_score', 'clarity_score', 'coverage_topk', 'shap_conservation_error']:
            if col in df.columns:
                summary['metrics'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                }
        
        # SHAP conservation
        if 'shap_conservation_valid' in df.columns:
            valid_count = df['shap_conservation_valid'].sum()
            summary['metrics']['shap_conservation_rate'] = float(valid_count / len(df)) if len(df) > 0 else 0.0
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            summary[f'dataset_{dataset}'] = {
                'count': len(dataset_df),
                'clarity_mean': float(dataset_df['clarity_score'].mean()),
                'coverage_mean': float(dataset_df['coverage_topk'].mean()),
                'fidelity_mean': float(dataset_df['fidelity_score'].mean()),
            }
        
        self.summary = summary
        return summary
    
    def export_results_csv(self, filename: Optional[str] = None) -> str:
        if not self.results:
            logger.warning("No results to export")
            return None
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        filepath = filename or f"{self.config.output_dir}/evaluation_results.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
        
        return filepath
    
    def export_summary_json(self, filename: Optional[str] = None) -> str:
        if not self.summary:
            self.aggregate_results()
        
        filepath = filename or f"{self.config.output_dir}/summary.json"
        with open(filepath, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        logger.info(f"Summary exported to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        if not self.summary:
            self.aggregate_results()
        
        report = f"""
XAI-NLG FRAMEWORK EVALUATION REPORT


Timestamp: {self.summary.get('timestamp', 'N/A')}
Total Instances Evaluated: {self.summary.get('total_instances', 0)}

Global metrics:
{'-' * 60}
"""
        
        metrics = self.summary.get('metrics', {})
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                report += f"""
{metric_name}:
  Mean:   {metric_data.get('mean', 0):.3f}
  Median: {metric_data.get('median', 0):.3f}
  Std:    {metric_data.get('std', 0):.3f}
  Range:  [{metric_data.get('min', 0):.3f}, {metric_data.get('max', 0):.3f}]
"""
            elif isinstance(metric_data, dict):
                report += f"\n{metric_name}:\n"
                for k, v in metric_data.items():
                    report += f"  {k}: {v}\n"
            else:
                report += f"\n{metric_name}: {metric_data}\n"
        
        # Per dataset
        report += f"\n\nDataset results:\n{'-' * 60}\n"
        for key, value in self.summary.items():
            if key.startswith('dataset_'):
                report += f"\n{key}:\n"
                for k, v in value.items():
                    report += f"  {k}: {v}\n"
        
        report += f"\n\nTresholds and interpretation:\n{'-' * 60}\n"
        report += """
Clarity Score (0-100):
  - [45-100]: Good
  - [60-100]: Very good
  - [80-100]: Excellent

Coverage Top-K (%):
  - [80-100]: Good (features are mentioned)
  - [60-80]:  Moderate (some features are missing)
  - [<60]:    Weak (too few features)

Fidelity Score (0-1):
  - [0.8-1.0]: Good (correct signs)
  - [0.6-0.8]: Moderate
  - [<0.6]:   Weak

SHAP Conservation:
  - [100%]:   Perfect
  - [95-100%]: Acceptable
  - [<95%]:   Needs investigation

Jaccard@K (robustness):
  - [0.6-1.0]: Robust
  - [0.4-0.6]: Moderate
  - [<0.4]:   Unstable
"""
        
        return report

if __name__ == '__main__':
    print("Evaluator module ready for import and use")