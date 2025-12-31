"""Drift Attribution: Classify drift as benign, suspicious, or hostile."""

import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DriftAttributor:
    """
    Classify drift as benign, suspicious, or hostile.
    
    Uses directedness metric to distinguish attacks from natural shifts.
    """
    
    def __init__(
        self,
        baseline_centroid: Optional[np.ndarray] = None,
        unsafe_centroid: Optional[np.ndarray] = None
    ):
        """
        Args:
            baseline_centroid: Mean of safe calibration data
            unsafe_centroid: Mean of known unsafe examples
        """
        self.baseline_centroid = baseline_centroid
        self.unsafe_centroid = unsafe_centroid
        
        logger.info("DriftAttributor initialized")
    
    def set_centroids(
        self,
        baseline: np.ndarray,
        unsafe: Optional[np.ndarray] = None
    ):
        """Set reference centroids."""
        self.baseline_centroid = baseline
        self.unsafe_centroid = unsafe
    
    def compute_directedness(self, current_window: np.ndarray) -> float:
        """
        Measure if drift is directional toward unsafe region.
        
        Args:
            current_window: Recent embeddings (n x d)
        
        Returns:
            Directedness score [0, 1]
            - Low (< 0.3) = random/benign
            - High (> 0.7) = goal-directed/attack
        """
        if self.baseline_centroid is None or self.unsafe_centroid is None:
            # No unsafe centroid → can't compute directedness
            return 0.0
        
        current_centroid = np.mean(current_window, axis=0)
        
        # Vector from baseline to current
        drift_vector = current_centroid - self.baseline_centroid
        
        # Vector from baseline to unsafe
        attack_vector = self.unsafe_centroid - self.baseline_centroid
        
        # Cosine similarity
        drift_norm = np.linalg.norm(drift_vector)
        attack_norm = np.linalg.norm(attack_vector)
        
        if drift_norm < 1e-8 or attack_norm < 1e-8:
            return 0.0
        
        directedness = np.dot(drift_vector, attack_vector) / (drift_norm * attack_norm)
        
        # Normalize to [0, 1]
        return float(max(0.0, min(1.0, (directedness + 1) / 2)))
    
    def classify_drift(
        self,
        mmd: float,
        directedness: float,
        velocity: float,
        window_size: int
    ) -> str:
        """
        Classify drift type.
        
        Args:
            mmd: Current MMD score
            directedness: Directedness toward unsafe [0, 1]
            velocity: Rate of change
            window_size: Number of samples
        
        Returns:
            "benign", "suspicious", or "hostile"
        """
        # Sudden + directional = attack
        if velocity > 0.05 and directedness > 0.7:
            logger.warning(
                f"Hostile drift detected: velocity={velocity:.4f}, "
                f"directedness={directedness:.2f}"
            )
            return "hostile"
        
        # High MMD but random = data quality issue
        if mmd > 0.15 and directedness < 0.3:
            logger.info("Suspicious drift: high MMD, low directedness")
            return "suspicious"
        
        # Slow monotonic = benign shift
        if velocity < 0.01 and mmd < 0.20:
            return "benign"
        
        # Default: based on directedness
        if directedness > 0.5:
            return "suspicious"
        else:
            return "benign"
    
    def get_stats(self, current_window: np.ndarray) -> Dict:
        """Get attribution statistics."""
        if len(current_window) == 0:
            return {
                "directedness": 0.0,
                "has_centroids": False
            }
        
        directedness = self.compute_directedness(current_window)
        
        return {
            "directedness": directedness,
            "has_centroids": self.unsafe_centroid is not None,
        }
