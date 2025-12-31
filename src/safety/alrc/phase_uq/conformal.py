"""Split Conformal Prediction for binary safety classification.

Implements label-stratified calibration to handle class imbalance.
"""

import numpy as np
from typing import Dict, Set, Tuple
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result from conformal predictor."""
    prediction_set: Set[str]
    is_uncertain: bool
    set_certainty: float  # 1.0 for singleton, 0.0 for uncertain
    tau_used: float
    model_version: str


class SplitConformalPredictor:
    """
    Split Conformal Prediction with stratified calibration.
    
    Provides coverage guarantee: P(Y ∈ prediction_set) ≥ 1 - α
    
    Uses per-label thresholds to handle class imbalance correctly.
    """
    
    LABELS = ["safe", "unsafe"]
    
    def __init__(self, alpha: float = 0.05, model_version: str = None):
        """
        Args:
            alpha: Error rate (0.05 = 95% coverage)
            model_version: Hash of model components for versioning
        """
        self.alpha = alpha
        self.tau_safe = None
        self.tau_unsafe = None
        self.calibrated = False
        self.model_version = model_version or "unknown"
        
        self.n_cal_safe = 0
        self.n_cal_unsafe = 0
    
    def calibrate_stratified(
        self, 
        safe_scores: np.ndarray, 
        unsafe_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute per-label thresholds (Mondrian CP style).
        
        This handles class imbalance correctly by calibrating
        each label separately.
        
        Args:
            safe_scores: Non-conformity scores for safe samples
            unsafe_scores: Non-conformity scores for unsafe samples
        
        Returns:
            Dict of thresholds per label
        """
        self.n_cal_safe = len(safe_scores)
        self.n_cal_unsafe = len(unsafe_scores)
        
        if self.n_cal_safe == 0 or self.n_cal_unsafe == 0:
            raise ValueError("Need samples from both classes")
        
        # Compute quantile index
        def quantile_index(n):
            return np.ceil((n + 1) * (1 - self.alpha)) / n
        
        q_safe = quantile_index(self.n_cal_safe)
        q_unsafe = quantile_index(self.n_cal_unsafe)
        
        self.tau_safe = float(np.quantile(safe_scores, min(q_safe, 1.0)))
        self.tau_unsafe = float(np.quantile(unsafe_scores, min(q_unsafe, 1.0)))
        
        self.calibrated = True
        
        logger.info(
            f"Calibrated SCP: α={self.alpha}, "
            f"τ_safe={self.tau_safe:.4f} (n={self.n_cal_safe}), "
            f"τ_unsafe={self.tau_unsafe:.4f} (n={self.n_cal_unsafe})"
        )
        
        return {"safe": self.tau_safe, "unsafe": self.tau_unsafe}
    
    def calibrate(self, scores: np.ndarray) -> float:
        """
        Standard non-stratified calibration (for backward compatibility).
        
        Warning: Use calibrate_stratified for production.
        """
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        tau = float(np.quantile(scores, min(q, 1.0)))
        
        self.tau_safe = tau
        self.tau_unsafe = tau
        self.calibrated = True
        
        logger.warning("Using non-stratified calibration - consider calibrate_stratified")
        
        return tau
    
    def predict_set(self, probs: Dict[str, float]) -> ConformalResult:
        """
        Predict label SET with coverage guarantee.
        
        Args:
            probs: {"safe": p_safe, "unsafe": p_unsafe}
        
        Returns:
            ConformalResult with prediction set
        """
        if not self.calibrated:
            raise RuntimeError("Must calibrate before prediction")
        
        pred_set = set()
        
        # Non-conformity score: s(x, y) = 1 - p(y|x)
        score_safe = 1.0 - probs.get("safe", 0.0)
        score_unsafe = 1.0 - probs.get("unsafe", 0.0)
        
        # Include label if score ≤ threshold
        if score_safe <= self.tau_safe:
            pred_set.add("safe")
        
        if score_unsafe <= self.tau_unsafe:
            pred_set.add("unsafe")
        
        # Edge case: empty set (extreme OOD)
        # Conservative: include both labels (maximum uncertainty)
        if not pred_set:
            pred_set = {"safe", "unsafe"}
            logger.warning("Empty prediction set - extreme OOD detected")
        
        is_uncertain = len(pred_set) > 1
        set_certainty = 1.0 if len(pred_set) == 1 else 0.0
        
        # Use max tau for logging (most conservative applied)
        tau_used = max(self.tau_safe, self.tau_unsafe)
        
        return ConformalResult(
            prediction_set=pred_set,
            is_uncertain=is_uncertain,
            set_certainty=set_certainty,
            tau_used=tau_used,
            model_version=self.model_version
        )
    
    def predict_from_risk(self, risk_score: float) -> ConformalResult:
        """
        Convenience: predict from risk score instead of probs.
        
        Args:
            risk_score: Value in [0, 1] where 1 = unsafe
        """
        probs = {
            "safe": 1.0 - risk_score,
            "unsafe": risk_score
        }
        return self.predict_set(probs)
    
    def save(self, path: str):
        """Save calibrated state with model version."""
        data = {
            'tau_safe': self.tau_safe,
            'tau_unsafe': self.tau_unsafe,
            'alpha': self.alpha,
            'calibrated': self.calibrated,
            'model_version': self.model_version,
            'n_cal_safe': self.n_cal_safe,
            'n_cal_unsafe': self.n_cal_unsafe,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved calibration to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SplitConformalPredictor':
        """Load calibrated predictor."""
        with open(path) as f:
            data = json.load(f)
        
        pred = cls(
            alpha=data['alpha'], 
            model_version=data.get('model_version', 'unknown')
        )
        pred.tau_safe = data['tau_safe']
        pred.tau_unsafe = data['tau_unsafe']
        pred.calibrated = data['calibrated']
        pred.n_cal_safe = data.get('n_cal_safe', 0)
        pred.n_cal_unsafe = data.get('n_cal_unsafe', 0)
        
        logger.info(f"Loaded calibration from {path}")
        
        return pred
    
    def get_stats(self) -> Dict:
        """Get calibration statistics."""
        return {
            'alpha': self.alpha,
            'tau_safe': self.tau_safe,
            'tau_unsafe': self.tau_unsafe,
            'calibrated': self.calibrated,
            'model_version': self.model_version,
            'n_cal_safe': self.n_cal_safe,
            'n_cal_unsafe': self.n_cal_unsafe,
        }
