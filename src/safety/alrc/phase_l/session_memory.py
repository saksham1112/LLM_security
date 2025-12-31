"""
SessionMemory: Hot-tier Redis storage for recent session embeddings.

Stores (embedding, risk, intent_profile) with automatic TTL expiry.
"""

import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
from redis import Redis
from redis.exceptions import RedisError
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    Redis-backed session memory with TTL-based privacy.
    
    Stores recent turns as append-only list with automatic expiry.
    """
    
    # Privacy: 30 min session expiry
    TTL_SECONDS = 1800
    
    # Max turns to store (prevent unbounded growth)
    MAX_TURNS = 100
    
    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        """
        Initialize session memory.
        
        Args:
            redis_url: Redis connection string
            db: Redis database number
        """
        try:
            self.redis = Redis.from_url(redis_url, db=db, decode_responses=False)
            self.redis.ping()
            logger.info(f"SessionMemory connected to Redis at {redis_url}")
        except RedisError as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None
    
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"
    
    def store_turn(
        self,
        session_id: str,
        embedding: np.ndarray,
        risk: float,
        intent_profile: Dict[str, float]
    ) -> bool:
        """
        Store a turn in session memory.
        
        Args:
            session_id: Session identifier (will be hashed)
            embedding: Embedding vector (384-dim)
            risk: Risk score [0, 1]
            intent_profile: Intent distribution
        
        Returns:
            True if stored successfully, False otherwise
        """
        if self.redis is None:
            return False
        
        try:
            key = self._session_key(session_id)
            
            # Serialize turn data
            turn_data = {
                'embedding': embedding.tobytes(),
                'risk': risk,
                'intent_profile': intent_profile,
                'shape': embedding.shape
            }
            serialized = pickle.dumps(turn_data)
            
            # Append to list
            self.redis.rpush(key, serialized)
            
            # Trim to MAX_TURNS (FIFO)
            self.redis.ltrim(key, -self.MAX_TURNS, -1)
            
            # Set/refresh TTL
            self.redis.expire(key, self.TTL_SECONDS)
            
            return True
            
        except RedisError as e:
            logger.error(f"Failed to store turn: {e}")
            return False
    
    def get_trajectory(self, session_id: str) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Retrieve full session trajectory.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of (embedding, risk, intent_profile) tuples
        """
        if self.redis is None:
            return []
        
        try:
            key = self._session_key(session_id)
            raw_turns = self.redis.lrange(key, 0, -1)
            
            trajectory = []
            for raw_turn in raw_turns:
                turn_data = pickle.loads(raw_turn)
                embedding = np.frombuffer(
                    turn_data['embedding'],
                    dtype=np.float32
                ).reshape(turn_data['shape'])
                trajectory.append((
                    embedding,
                    turn_data['risk'],
                    turn_data['intent_profile']
                ))
            
            return trajectory
            
        except (RedisError, pickle.PickleError) as e:
            logger.error(f"Failed to retrieve trajectory: {e}")
            return []
    
    def get_last_embedding(self, session_id: str) -> Optional[np.ndarray]:
        """Get most recent embedding from session."""
        trajectory = self.get_trajectory(session_id)
        if trajectory:
            return trajectory[-1][0]
        return None
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session has stored data."""
        if self.redis is None:
            return False
        try:
            return self.redis.exists(self._session_key(session_id)) > 0
        except RedisError:
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Manually delete session (for testing or privacy requests)."""
        if self.redis is None:
            return False
        try:
            self.redis.delete(self._session_key(session_id))
            return True
        except RedisError as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions (for monitoring)."""
        if self.redis is None:
            return 0
        try:
            return len(self.redis.keys("session:*"))
        except RedisError:
            return 0
