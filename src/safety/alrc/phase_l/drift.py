"""
DriftTracker: Compute long-term drift toward hazard regions.

Tracks accumulated drift over time using leaky integration.
"""

from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DriftTracker:
    """
    Tracks long-term semantic drift toward hazard centroids.
    
    Uses leaky integration to accumulate positive drift (movement toward harm)
    while decaying past drift signals.
    """
    
    def __init__(self, hazard_centroids: List[np.ndarray], decay: float = 0.9):
        """
        Initialize drift tracker.
        
        Args:
            hazard_centroids: List of hazard prototype vectors
            decay: Decay factor for leaky integration (0.9 = 10% decay per turn)
        """
        self.hazard_centroids = [c.astype(np.float32) for c in hazard_centroids]
        self.decay = decay
        
        # Accumulated drift
        self.drift_accumulator: float = 0.0
        
        # Previous similarity (for delta calculation)
        self.previous_similarity: float = 0.0
    
    def compute_drift(self, prototype: np.ndarray) -> float:
        """
        Compute drift toward nearest hazard centroid.
        
        Algorithm:
            sim_current = max(sim(prototype, hazard_i))
            drift = sim_current - sim_previous
            drift_acc = decay * drift_acc + max(0, drift)
        
        Args:
            prototype: Current user prototype vector
        
        Returns:
            Accumulated drift score
        """
        if not self.hazard_centroids:
            return 0.0
        
        # Find max similarity to any hazard centroid
        similarities = [
            self._cosine_similarity(prototype, hazard)
            for hazard in self.hazard_centroids
        ]
        current_similarity = max(similarities)
        
        # Calculate drift (change in similarity)
        drift_delta = current_similarity - self.previous_similarity
        
        # Only accumulate positive drift (movement toward hazard)
        positive_drift = max(0.0, drift_delta)
        
        # Leaky integration: decay old drift, add new
        self.drift_accumulator = self.decay * self.drift_accumulator + positive_drift
        
        # Update previous for next iteration
        self.previous_similarity = current_similarity
        
        logger.debug(
            f"Drift: sim={current_similarity:.3f}, delta={drift_delta:.3f}, "
            f"acc={self.drift_accumulator:.3f}"
        )
        
        return self.drift_accumulator
    
    def reset(self):
        """Reset drift tracker (e.g., on session reset)."""
        self.drift_accumulator = 0.0
        self.previous_similarity = 0.0
    
    def get_drift_velocity(self) -> float:
        """
        Get instantaneous drift velocity (for monitoring).
        
        Returns:
            Rate of change in similarity
        """
        return self.previous_similarity
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
