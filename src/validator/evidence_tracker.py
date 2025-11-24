from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class EvidenceRecord:
    """Record linking text to XAI values."""
    timestamp: str
    statement: str
    method: str  # SHAP, LIME
    features: List[str]
    contributions: List[float]
    fidelity_score: float
    validation_status: Dict[str, Any]

class EvidenceTracker:
    def __init__(self):
        self.records: List[EvidenceRecord] = []
    
    def add_record(
        self,
        statement: str,
        method: str,
        features: List[str],
        contributions: List[float],
        fidelity_score: float,
        validation_status: Dict[str, Any] = None
    ):
        """
        Add evidence record.
        
        Args:
            statement: Generated text
            method: XAI method used
            features: List of features
            contributions: List of contributions
            fidelity_score: Alignment score
            validation_status: Validation results
        """
        record = EvidenceRecord(
            timestamp=datetime.now().isoformat(),
            statement=statement,
            method=method,
            features=features,
            contributions=contributions,
            fidelity_score=fidelity_score,
            validation_status=validation_status or {}
        )
        self.records.append(record)
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        
        data = [asdict(record) for record in self.records]
        return pd.DataFrame(data)
    
    def get_traceability_table(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        
        return df[[
            "timestamp",
            "statement",
            "method",
            "features",
            "contributions",
            "fidelity_score"
        ]]
    
    def export_csv(self, filepath: str):
        self.to_dataframe().to_csv(filepath, index=False)
