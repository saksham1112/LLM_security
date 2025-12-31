"""
LongTermMemory: Integration layer for Phase L.

Orchestrates SessionMemory, UserPrototype, and DriftTracker.
Provides simple interface for Phase T integration.
"""

from typing import Dict, Optional
import numpy as np
import logging
import hashlib

from .session_memory import SessionMemory
from .prototype import UserPrototype
from .drift import DriftTracker

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Phase L orchestrator.
    
    Manages:
    - Session memory (Redis hot tier)
    - User prototypes (DPU)
    - Long-term drift tracking
    
    Fail-safe: If any component fails, returns neutral signals.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        hazard_centroids: Optional[np.ndarray] = None,
        embedding_dim: int = 384,
        alpha: float = 0.05,
        decay: float = 0.9,
        enabled: bool = True
    ):
        """
        Initialize long-term memory.
        
        Args:
            redis_url: Redis connection string
            hazard_centroids: Hazard prototype vectors for drift detection
            embedding_dim: Embedding dimensionality
            alpha: Prototype learning rate
            decay: Drift decay factor
            enabled: Master switch (for A/B testing)
        """
        self.enabled = enabled
        self.embedding_dim = embedding_dim
        
        if not enabled:
            logger.info("Phase L disabled")
            return
        
        # Initialize components
        self.session_memory = SessionMemory(redis_url=redis_url)
        
        # User prototypes cache (in-memory)
        self._prototypes: Dict[str, UserPrototype] = {}
        self.alpha = alpha
        
        # Drift tracker
        if hazard_centroids is None:
            hazard_centroids = []
        self.drift_tracker_template = DriftTracker(
            hazard_centroids=hazard_centroids,
            decay=decay
        )
        
        # Per-session drift trackers
        self._drift_trackers: Dict[str, DriftTracker] = {}
        
        logger.info(f"Phase L initialized (embedding_dim={embedding_dim}, alpha={alpha})")
    
    def update(
        self,
        session_id: str,
        embedding: np.ndarray,
        risk: float,
        intent_profile: Dict[str, float]
    ) -> float:
        """
        Update long-term memory and return drift signal.
        
        Args:
            session_id: Session identifier (will be hashed)
            embedding: Current turn embedding
            risk: Current risk score
            intent_profile: Current intent distribution
        
        Returns:
            long_term_drift: Accumulated drift score [0, inf)
        """
        if not self.enabled:
            return 0.0
        
        try:
            # Hash session ID for privacy
            session_hash = self._hash_session_id(session_id)
            
            # Store in session memory (Redis)
            self.session_memory.store_turn(
                session_hash,
                embedding,
                risk,
                intent_profile
            )
            
            # Update user prototype
            if session_hash not in self._prototypes:
                self._prototypes[session_hash] = UserPrototype(
                    embedding_dim=self.embedding_dim,
                    alpha=self.alpha
                )
            
            prototype, variance = self._prototypes[session_hash].update(embedding)
            
            # Compute drift
            if session_hash not in self._drift_trackers:
                self._drift_trackers[session_hash] = DriftTracker(
                    hazard_centroids=self.drift_tracker_template.hazard_centroids,
                    decay=self.drift_tracker_template.decay
                )
            
            drift_score = self._drift_trackers[session_hash].compute_drift(prototype)
            
            logger.debug(
                f"Phase L: session={session_hash[:8]}, drift={drift_score:.3f}, "
                f"variance={variance:.3f}"
            )
            
            return drift_score
            
        except Exception as e:
            logger.error(f"Phase L update failed: {e}")
            return 0.0  # Fail-safe: neutral signal
    
    def get_prototype(self, session_id: str) -> Optional[np.ndarray]:
        """Get current prototype for session."""
        session_hash = self._hash_session_id(session_id)
        if session_hash in self._prototypes:
            return self._prototypes[session_hash].prototype
        return None
    
    def reset_session(self, session_id: str):
        """Reset session state (memory + prototype + drift)."""
        if not self.enabled:
            return
        
        session_hash = self._hash_session_id(session_id)
        
        # Clear Redis
        self.session_memory.delete_session(session_hash)
        
        # Clear in-memory caches
        if session_hash in self._prototypes:
            del self._prototypes[session_hash]
        if session_hash in self._drift_trackers:
            del self._drift_trackers[session_hash]
        
        logger.debug(f"Reset session: {session_hash[:8]}")
    
    def get_stats(self) -> Dict:
        """Get Phase L statistics (for monitoring)."""
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'active_sessions': self.session_memory.get_active_sessions_count(),
            'cached_prototypes': len(self._prototypes),
            'cached_trackers': len(self._drift_trackers),
        }
    
    @staticmethod
    def _hash_session_id(session_id: str) -> str:
        """Hash session ID for privacy."""
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]
