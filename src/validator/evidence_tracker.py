"""
Evidence Tracker for maintaining audit trail of explanations.
"""
import csv
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class EvidenceRecord:
    """Record of a single explanation for audit purposes."""
    timestamp: str
    instance_id: str
    method: str  # "SHAP" or "LIME"
    prediction: str
    top_features: List[str]
    contributions: List[float]
    generated_text: str
    nlg_technique: str
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceTracker:
    """
    Tracks and stores evidence for all generated explanations.
    
    Provides audit trail linking:
    - XAI contributions to generated text
    - Validation results to explanations
    - Timestamps and instance identifiers
    """
    
    def __init__(self):
        """Initialize empty evidence tracker."""
        self.records: List[EvidenceRecord] = []
    
    def add_record(
        self,
        instance_id: str,
        method: str,
        prediction: str,
        contributions: Dict[str, float],
        generated_text: str,
        nlg_technique: str,
        validation_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvidenceRecord:
        """
        Add a new evidence record.
        
        Args:
            instance_id: Unique identifier for the instance
            method: XAI method used ("SHAP" or "LIME")
            prediction: Model prediction as string
            contributions: Feature contribution dictionary
            generated_text: Generated NLG explanation
            nlg_technique: NLG technique used
            validation_results: Validation check results
            metadata: Additional metadata
            
        Returns:
            Created EvidenceRecord
        """
        # Rank features by absolute contribution
        ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        record = EvidenceRecord(
            timestamp=datetime.now().isoformat(),
            instance_id=str(instance_id),
            method=method.upper(),
            prediction=prediction,
            top_features=[f for f, _ in ranked[:10]],
            contributions=[v for _, v in ranked[:10]],
            generated_text=generated_text,
            nlg_technique=nlg_technique,
            validation_results=validation_results,
            metadata=metadata or {}
        )
        
        self.records.append(record)
        return record
    
    def get_records(
        self,
        method: Optional[str] = None,
        technique: Optional[str] = None,
        valid_only: bool = False
    ) -> List[EvidenceRecord]:
        """
        Get filtered records.
        
        Args:
            method: Filter by XAI method
            technique: Filter by NLG technique
            valid_only: Only return records that passed validation
            
        Returns:
            Filtered list of records
        """
        records = self.records
        
        if method:
            records = [r for r in records if r.method.upper() == method.upper()]
        
        if technique:
            records = [r for r in records if r.nlg_technique == technique]
        
        if valid_only:
            records = [r for r in records if r.validation_results.get("valid", False)]
        
        return records
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert records to pandas DataFrame.
        
        Returns:
            DataFrame with one row per record
        """
        if not self.records:
            return pd.DataFrame()
        
        data = []
        for r in self.records:
            row = {
                "timestamp": r.timestamp,
                "instance_id": r.instance_id,
                "method": r.method,
                "prediction": r.prediction,
                "nlg_technique": r.nlg_technique,
                "top_feature_1": r.top_features[0] if r.top_features else None,
                "top_feature_2": r.top_features[1] if len(r.top_features) > 1 else None,
                "top_feature_3": r.top_features[2] if len(r.top_features) > 2 else None,
                "contribution_1": r.contributions[0] if r.contributions else None,
                "contribution_2": r.contributions[1] if len(r.contributions) > 1 else None,
                "contribution_3": r.contributions[2] if len(r.contributions) > 2 else None,
                "clarity_score": r.validation_results.get("clarity", {}).get("score"),
                "coverage_score": r.validation_results.get("coverage", {}).get("coverage_score"),
                "valid": r.validation_results.get("valid", False),
                "text_preview": r.generated_text[:100] + "..." if len(r.generated_text) > 100 else r.generated_text
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_csv(self, filepath: str):
        """
        Export records to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def export_json(self, filepath: str):
        """
        Export records to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        data = [asdict(r) for r in self.records]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of all records.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.records:
            return {"total_records": 0}
        
        df = self.to_dataframe()
        
        return {
            "total_records": len(self.records),
            "by_method": df["method"].value_counts().to_dict(),
            "by_technique": df["nlg_technique"].value_counts().to_dict(),
            "avg_clarity_score": df["clarity_score"].mean(),
            "avg_coverage_score": df["coverage_score"].mean(),
            "valid_percentage": df["valid"].mean() * 100
        }