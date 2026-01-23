"""Data drift detection module using Evidently."""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift in input features."""
    
    def __init__(self, reference_data: pd.DataFrame, column_names: List[str] = None):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: DataFrame with reference data for comparison
            column_names: List of column names (if not in DataFrame)
        """
        self.reference_data = reference_data
        self.column_names = column_names or [f"feature_{i}" for i in range(reference_data.shape[1])]
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: DataFrame with current data
            
        Returns:
            Dictionary with drift detection results
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        drift_results = report.as_dict()
        logger.info(f"Drift detection completed: {drift_results}")
        
        return drift_results