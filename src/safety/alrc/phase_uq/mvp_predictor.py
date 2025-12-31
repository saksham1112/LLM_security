"""Multivalid Prediction with group-conditional coverage."""

import numpy as np
from typing import Dict, Set
from dataclasses import dataclass
import logging

from .conformal import ConformalResult
from .mvp_groups import GROUP_DEFINITIONS, get_group_threshold

logger = logging.getLogger(__name__)


class MVPPredictor:
    """
    Multivalid Prediction for group-conditional coverage.
    
    Maintains per-group thresholds to ensure coverage even when
    adversary targets specific subgroups.
    
    Key property: Coverage holds for each group independently,
    not just on average (marginal).
    """
    
    def __init__(self, model_version: str = None):
        """
        Args:
            model_version: Hash of model components
        """
        self.model_version = model_version or "unknown"
        
        # Per-group thresholds (loaded from group definitions)
        self.thresholds = {
            group: info["threshold"]
            for group, info in GROUP_DEFINITIONS.items()
        }
        
        self.calibrated = True  # Thresholds predefined
        
        logger.info(f"Initialized MVP with {len(self.thresholds)} groups")
    
    def predict_set(
        self, 
        probs: Dict[str, float], 
        groups: Set[str]
    ) -> ConformalResult:
        """
        Predict set using strictest group threshold.
        
        CRITICAL: Uses min() threshold (stricter).
        
        Args:
            probs: {"safe": p_safe, "unsafe": p_unsafe}
            groups: Active group names
        
        Returns:
            ConformalResult with prediction set
        """
        # Get strictest threshold
        tau = get_group_threshold(groups)
        
        pred_set = set()
        
        # Non-conformity scores
        score_safe = 1.0 - probs.get("safe", 0.0)
        score_unsafe = 1.0 - probs.get("unsafe", 0.0)
        
        # Include if score ≤ threshold
        if score_safe <= tau:
            pred_set.add("safe")
        
        if score_unsafe <= tau:
            pred_set.add("unsafe")
        
        # Edge case: empty set
        if not pred_set:
            pred_set = {"safe", "unsafe"}
            logger.warning(f"MVP: Empty set for groups {groups}, τ={tau:.3f}")
        
        is_uncertain = len(pred_set) > 1
        set_certainty = 1.0 if len(pred_set) == 1 else 0.0
        
        return ConformalResult(
            prediction_set=pred_set,
            is_uncertain=is_uncertain,
            set_certainty=set_certainty,
            tau_used=tau,
            model_version=self.model_version
        )
    
    def predict_from_risk(
        self, 
        risk_score: float, 
        groups: Set[str]
    ) -> ConformalResult:
        """Convenience: predict from risk score."""
        probs = {
            "safe": 1.0 - risk_score,
            "unsafe": risk_score
        }
        return self.predict_set(probs, groups)
    
    def get_stats(self) -> Dict:
        """Get MVP statistics."""
        return {
            'model_version': self.model_version,
            'num_groups': len(self.thresholds),
            'thresholds': self.thresholds,
        }
