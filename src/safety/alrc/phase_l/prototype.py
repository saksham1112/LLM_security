"""
UserPrototype: Dynamic Prototype Updating (DPU)

Maintains a running prototype vector per user/session using momentum-based updates.
Detects long-term intent drift.
"""

from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UserPrototype:
    """
    Momentum-based prototype for detecting long-term intent drift.
    
    Maintains:
    - Prototype vector p (centroid of user's embeddings)
    - Variance σ² (spread)
    - Count n (number of updates)
    """
    
    def __init__(self, embedding_dim: int = 384, alpha: float = 0.05):
        """
        Initialize user prototype.
        
        Args:
            embedding_dim: Dimensionality of embeddings
            alpha: Learning rate (0.05 = 5% weight to new data)
        """
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Prototype state
        self.prototype: np.ndarray = np.zeros(embedding_dim, dtype=np.float32)
        self.variance: float = 0.0
        self.count: int = 0
        
        # Track previous prototype for drift calculation
        self.previous_prototype: np.ndarray = np.zeros(embedding_dim, dtype=np.float32)
    
    def update(self, embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update prototype with new embedding.
        
        Algorithm:
            p ← (1 - α) p + α x
            σ² ← (1 - α) σ² + α ||x - p||²
            n ← n + 1
        
        Args:
            embedding: New embedding vector
        
        Returns:
            (updated_prototype, variance)
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embedding.shape[0]} != expected {self.embedding_dim}"
            )
        
        # First update: initialize prototype
        if self.count == 0:
            self.prototype = embedding.copy().astype(np.float32)
            self.previous_prototype = self.prototype.copy()
            self.variance = 0.0
            self.count = 1
            return self.prototype, self.variance
        
        # Save previous for drift calculation
        self.previous_prototype = self.prototype.copy()
        
        # Momentum update of prototype
        self.prototype = (
            (1 - self.alpha) * self.prototype + 
            self.alpha * embedding.astype(np.float32)
        )
        
        # Update variance (measure of consistency)
        distance_sq = np.sum((embedding - self.prototype) ** 2)
        self.variance = (1 - self.alpha) * self.variance + self.alpha * distance_sq
        
        self.count += 1
        
        return self.prototype, self.variance
    
    def get_drift_magnitude(self) -> float:
        """
        Calculate magnitude of drift from previous to current prototype.
        
        Returns:
            Euclidean distance between previous and current prototype
        """
        if self.count <= 1:
            return 0.0
        
        return float(np.linalg.norm(self.prototype - self.previous_prototype))
    
    def get_stability_score(self) -> float:
        """
        Calculate stability score (inverse of variance).
        
        Returns:
            Stability in [0, 1] where 1 = very stable, 0 = very unstable
        """
        if self.count == 0:
            return 1.0
        
        # Normalize variance to stability score
        # Low variance = high stability
        return 1.0 / (1.0 + self.variance)
    
    def is_oscillating(self, threshold: float = 0.5) -> bool:
        """
        Detect if user is oscillating (probing attack signature).
        
        Args:
            threshold: Variance threshold for oscillation
        
        Returns:
            True if variance exceeds threshold (unstable behavior)
        """
        return self.variance > threshold
    
    def to_dict(self) -> Dict:
        """Serialize prototype state (for persistence)."""
        return {
            'prototype': self.prototype.tolist(),
            'previous_prototype': self.previous_prototype.tolist(),
            'variance': float(self.variance),
            'count': int(self.count),
            'alpha': float(self.alpha),
        }
    
    @classmethod
    def from_dict(cls, data: Dict, embedding_dim: int = 384) -> 'UserPrototype':
        """Deserialize prototype state."""
        instance = cls(embedding_dim=embedding_dim, alpha=data['alpha'])
        instance.prototype = np.array(data['prototype'], dtype=np.float32)
        instance.previous_prototype = np.array(data['previous_prototype'], dtype=np.float32)
        instance.variance = float(data['variance'])
        instance.count = int(data['count'])
        return instance
