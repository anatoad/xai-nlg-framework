"""
XAI-NLG Framework - Evaluator Module
Provides evaluation functionality for the XAI-NLG pipeline.
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    num_instances: int = 20
    num_robustness_runs: int = 5
    datasets: List[str] = field(default_factory=lambda: ['breast_cancer'])
    methods: List[str] = field(default_factory=lambda: ['shap', 'lime'])
    techniques: List[str] = field(default_factory=lambda: ['few_shot', 'cot', 'self_consistency'])
    top_k: int = 5
    output_dir: str = './evaluation_results'


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    instance_id: str
    model_name: str
    xai_method: str
    nlg_technique: str
    prediction: int
    true_label: int
    clarity_score: float
    coverage_score: float
    is_valid: bool
    generated_text: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'instance_id': self.instance_id,
            'model_name': self.model_name,
            'xai_method': self.xai_method,
            'nlg_technique': self.nlg_technique,
            'prediction': self.prediction,
            'true_label': self.true_label,
            'clarity_score': self.clarity_score,
            'coverage_score': self.coverage_score,
            'is_valid': self.is_valid,
            'text_length': len(self.generated_text),
            'error': self.error
        }


class Evaluator:
    """
    Evaluator for XAI-NLG Framework.
    Evaluates explanations across multiple dimensions.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results: List[EvaluationResult] = []
        self.summary = {}
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_single(
        self,
        pipeline,
        instance: np.ndarray,
        true_label: int,
        instance_id: str,
        model_name: str,
        xai_method: str,
        nlg_technique: str
    ) -> EvaluationResult:
        """
        Evaluate a single instance.
        """
        try:
            # Get explanation from pipeline
            result = pipeline.explain_instance(
                instance=instance,
                method=xai_method,
                technique=nlg_technique,
                generate_text=True
            )
            
            # Extract metrics from validation
            validation = result.get('validation', {})
            clarity = validation.get('clarity', {}).get('score', 0.0)
            coverage = validation.get('coverage', {}).get('coverage_score', 0.0) * 100
            is_valid = validation.get('valid', False)
            
            prediction = result.get('prediction', -1)
            generated_text = result.get('generated_text', '')
            
            eval_result = EvaluationResult(
                instance_id=instance_id,
                model_name=model_name,
                xai_method=xai_method,
                nlg_technique=nlg_technique,
                prediction=int(prediction),
                true_label=int(true_label),
                clarity_score=float(clarity),
                coverage_score=float(coverage),
                is_valid=bool(is_valid),
                generated_text=generated_text
            )
            
            self.results.append(eval_result)
            return eval_result
            
        except Exception as e:
            logger.error(f"Error evaluating {instance_id}: {str(e)}")
            eval_result = EvaluationResult(
                instance_id=instance_id,
                model_name=model_name,
                xai_method=xai_method,
                nlg_technique=nlg_technique,
                prediction=-1,
                true_label=int(true_label),
                clarity_score=0.0,
                coverage_score=0.0,
                is_valid=False,
                error=str(e)
            )
            self.results.append(eval_result)
            return eval_result
    
    def evaluate_batch(
        self,
        pipeline,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        xai_method: str,
        nlg_technique: str,
        num_instances: int = None
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of instances.
        """
        num_instances = num_instances or self.config.num_instances
        num_instances = min(num_instances, len(X))
        
        batch_results = []
        
        for i in range(num_instances):
            instance_id = f"{model_name}_{xai_method}_{nlg_technique}_{i}"
            
            result = self.evaluate_single(
                pipeline=pipeline,
                instance=X[i],
                true_label=y[i],
                instance_id=instance_id,
                model_name=model_name,
                xai_method=xai_method,
                nlg_technique=nlg_technique
            )
            
            batch_results.append(result)
            logger.info(f"  {instance_id}: Clarity={result.clarity_score:.1f}, Coverage={result.coverage_score:.0f}%")
        
        return batch_results
    
    def aggregate_results(self) -> Dict:
        """Aggregate all results into summary statistics."""
        if not self.results:
            return {}
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Filter successful evaluations
        valid_df = df[df['error'] == '']
        
        self.summary = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(df),
            'successful_evaluations': len(valid_df),
            'failed_evaluations': len(df) - len(valid_df),
            'overall': {
                'clarity_mean': float(valid_df['clarity_score'].mean()) if len(valid_df) > 0 else 0,
                'clarity_std': float(valid_df['clarity_score'].std()) if len(valid_df) > 0 else 0,
                'coverage_mean': float(valid_df['coverage_score'].mean()) if len(valid_df) > 0 else 0,
                'coverage_std': float(valid_df['coverage_score'].std()) if len(valid_df) > 0 else 0,
                'valid_rate': float(valid_df['is_valid'].mean() * 100) if len(valid_df) > 0 else 0
            },
            'by_xai_method': {},
            'by_nlg_technique': {},
            'by_model': {}
        }
        
        # By XAI method
        for method in valid_df['xai_method'].unique():
            method_df = valid_df[valid_df['xai_method'] == method]
            self.summary['by_xai_method'][method] = {
                'clarity_mean': float(method_df['clarity_score'].mean()),
                'coverage_mean': float(method_df['coverage_score'].mean()),
                'valid_rate': float(method_df['is_valid'].mean() * 100)
            }
        
        # By NLG technique
        for tech in valid_df['nlg_technique'].unique():
            tech_df = valid_df[valid_df['nlg_technique'] == tech]
            self.summary['by_nlg_technique'][tech] = {
                'clarity_mean': float(tech_df['clarity_score'].mean()),
                'coverage_mean': float(tech_df['coverage_score'].mean()),
                'valid_rate': float(tech_df['is_valid'].mean() * 100)
            }
        
        # By model
        for model in valid_df['model_name'].unique():
            model_df = valid_df[valid_df['model_name'] == model]
            self.summary['by_model'][model] = {
                'clarity_mean': float(model_df['clarity_score'].mean()),
                'coverage_mean': float(model_df['coverage_score'].mean()),
                'valid_rate': float(model_df['is_valid'].mean() * 100)
            }
        
        return self.summary
    
    def export_csv(self, filename: str = None) -> str:
        """Export results to CSV."""
        df = pd.DataFrame([r.to_dict() for r in self.results])
        filepath = filename or f"{self.config.output_dir}/evaluation_results.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")
        return filepath
    
    def export_json(self, filename: str = None) -> str:
        """Export summary to JSON."""
        if not self.summary:
            self.aggregate_results()
        
        filepath = filename or f"{self.config.output_dir}/summary.json"
        with open(filepath, 'w') as f:
            json.dump(self.summary, f, indent=2)
        logger.info(f"Summary exported to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """Generate text report."""
        if not self.summary:
            self.aggregate_results()
        
        report = f"""
================================================================================
                    XAI-NLG FRAMEWORK - EVALUATION REPORT
================================================================================

Timestamp: {self.summary.get('timestamp', 'N/A')}
Total Evaluations: {self.summary.get('total_evaluations', 0)}
Successful: {self.summary.get('successful_evaluations', 0)}
Failed: {self.summary.get('failed_evaluations', 0)}

OVERALL METRICS
---------------
Clarity Score:  Mean={self.summary['overall']['clarity_mean']:.1f}, Std={self.summary['overall']['clarity_std']:.1f}
Coverage Score: Mean={self.summary['overall']['coverage_mean']:.1f}%
Valid Rate:     {self.summary['overall']['valid_rate']:.1f}%

BY XAI METHOD
-------------
"""
        for method, stats in self.summary.get('by_xai_method', {}).items():
            report += f"{method.upper():>10}: Clarity={stats['clarity_mean']:.1f}, Coverage={stats['coverage_mean']:.1f}%, Valid={stats['valid_rate']:.0f}%\n"
        
        report += "\nBY NLG TECHNIQUE\n----------------\n"
        for tech, stats in self.summary.get('by_nlg_technique', {}).items():
            report += f"{tech:>18}: Clarity={stats['clarity_mean']:.1f}, Coverage={stats['coverage_mean']:.1f}%, Valid={stats['valid_rate']:.0f}%\n"
        
        report += "\nBY MODEL\n--------\n"
        for model, stats in self.summary.get('by_model', {}).items():
            report += f"{model:>20}: Clarity={stats['clarity_mean']:.1f}, Coverage={stats['coverage_mean']:.1f}%, Valid={stats['valid_rate']:.0f}%\n"
        
        report += """
================================================================================
                              END OF REPORT
================================================================================
"""
        return report


if __name__ == '__main__':
    print("Evaluator module ready for import")