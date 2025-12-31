"""MMD Drift Detector: Maximum Mean Discrepancy for distribution shift detection."""

import numpy as np
from typing import List, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MMDDriftDetector:
    """
    Maximum Mean Discrepancy drift detector.
    
    Detects distribution shifts in embedding space using simple MMD.
    """
    
    def __init__(
        self,
        baseline_embeddings: np.ndarray,
        window_size: int = 1000,
        threshold: float = 0.15,
        min_samples: int = 100
    ):
        """
        Args:
            baseline_embeddings: Training/calibration embeddings (n x d)
            window_size: Sliding window size for live traffic
            threshold: Alert threshold for MMD
            min_samples: Minimum samples before computing MMD
        """
        self.baseline_mean = np.mean(baseline_embeddings, axis=0)
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Sliding window
        self.live_window = deque(maxlen=window_size)
        
        # Statistics
        self.drift_history: List[float] = []
        self.alert_count = 0
        self.last_update = 0.0
        
        logger.info(
            f"MMDDriftDetector initialized: baseline_dim={len(self.baseline_mean)}, "
            f"window={window_size}, threshold={threshold}"
        )
    
    def update(self, embedding: np.ndarray) -> float:
        """
        Add new embedding and compute MMD.
        
        Returns:
            Current MMD score
        """
        import time
        
        self.live_window.append(embedding)
        self.last_update = time.time()
        
        if len(self.live_window) < self.min_samples:
            return 0.0  # Not enough data
        
        # Compute live mean
        live_mean = np.mean(self.live_window, axis=0)
        
        # MMD (simplified: L2 distance between means)
        mmd = np.linalg.norm(live_mean - self.baseline_mean)
        
        self.drift_history.append(mmd)
        
        # Alert if threshold exceeded
        if mmd > self.threshold:
            self.alert_count += 1
            logger.warning(f"Drift detected: MMD={mmd:.3f} > {self.threshold}")
        
        return mmd
    
    def reset_window(self):
        """Reset sliding window (e.g., after attack handled)."""
        logger.info("Resetting drift window")
        self.live_window.clear()
    
    def get_velocity(self) -> float:
        """
        Compute rate of change in MMD.
        
        Returns:
            Drift velocity (change per sample)
        """
        if len(self.drift_history) < 10:
            return 0.0
        
        recent = self.drift_history[-10:]
        # Simple linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        velocity = np.polyfit(x, y, 1)[0]
        
        return float(velocity)
    
    def get_stats(self) -> Dict:
        """Get drift statistics."""
        if not self.drift_history:
            return {
                "mmd_mean": 0.0,
                "mmd_max": 0.0,
                "mmd_current": 0.0,
                "alert_count": 0,
                "window_size": 0,
                "velocity": 0.0
            }
        
        return {
            "mmd_mean": float(np.mean(self.drift_history[-100:])),
            "mmd_max": float(np.max(self.drift_history[-100:])),
            "mmd_current": float(self.drift_history[-1]),
            "alert_count": self.alert_count,
            "window_size": len(self.live_window),
            "velocity": self.get_velocity(),
        }
