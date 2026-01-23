"""Monitoring and drift detection for deployed model."""
import logging
from typing import List, Dict

import pandas as pd
from src.mlops_loadconsumption.drift import DriftDetector

logger = logging.getLogger(__name__)


def check_drift_from_logs(reference_logs: List[Dict], current_logs: List[Dict]) -> Dict:
    """
    Check for data drift using API prediction logs.
    
    Args:
        reference_logs: List of reference prediction logs
        current_logs: List of current prediction logs
        
    Returns:
        Drift detection results
    """
    reference_df = pd.DataFrame(reference_logs)
    current_df = pd.DataFrame(current_logs)
    
    detector = DriftDetector(reference_df)
    drift_results = detector.detect_drift(current_df)
    
    logger.info(f"Drift detection results: {drift_results}")
    
    return drift_results