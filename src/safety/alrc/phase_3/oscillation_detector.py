"""Oscillation Detector: Detect automated attack probing (AutoAdv-style)."""

from typing import Dict
import logging

try:
    from redis import Redis
except ImportError:
    Redis = None

logger = logging.getLogger(__name__)


class OscillationDetector:
    """
    Detect automated attack probing.
    
    Uses Redis for distributed rate limiting.
    """
    
    REFUSAL_THRESHOLD = 5      # Max refusals in window
    WINDOW_SECONDS = 60        # Time window
    LOCKOUT_SECONDS = 300      # Lockout duration (5 min)
    
    def __init__(self, redis_url: str = "redis://localhost:6379", enabled: bool = True):
        self.enabled = enabled and Redis is not None
        
        if self.enabled:
            try:
                self.redis = Redis.from_url(redis_url, decode_responses=True)
                self.redis.ping()
                logger.info(f"OscillationDetector initialized with Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Oscillation detection disabled.")
                self.enabled = False
                self.redis = None
        else:
            self.redis = None
            logger.info("OscillationDetector disabled (Redis not available)")
    
    def track_refusal(self, session_id: str) -> bool:
        """
        Track a refusal for this session.
        
        Returns:
            True if oscillation detected (should lock out)
        """
        if not self.enabled:
            return False
        
        key = f"refusal:{session_id}"
        
        try:
            # Increment counter
            count = self.redis.incr(key)
            
            # Set TTL on first increment
            if count == 1:
                self.redis.expire(key, self.WINDOW_SECONDS)
            
            # Check threshold
            if count >= self.REFUSAL_THRESHOLD:
                logger.warning(
                    f"Oscillation detected: session={session_id[:8]}, "
                    f"refusals={count} in {self.WINDOW_SECONDS}s"
                )
                
                # Set lockout
                lockout_key = f"lockout:{session_id}"
                self.redis.setex(lockout_key, self.LOCKOUT_SECONDS, "1")
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Oscillation tracking failed: {e}")
            return False
    
    def is_locked_out(self, session_id: str) -> bool:
        """Check if session is currently locked out."""
        if not self.enabled:
            return False
        
        try:
            lockout_key = f"lockout:{session_id}"
            return self.redis.exists(lockout_key) > 0
        except Exception as e:
            logger.error(f"Lockout check failed: {e}")
            return False
    
    def reset_session(self, session_id: str):
        """Reset counters for session."""
        if not self.enabled:
            return
        
        try:
            self.redis.delete(f"refusal:{session_id}")
            self.redis.delete(f"lockout:{session_id}")
            logger.info(f"Reset oscillation counters for session={session_id[:8]}")
        except Exception as e:
            logger.error(f"Session reset failed: {e}")
    
    def get_stats(self, session_id: str) -> Dict:
        """Get statistics for session."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            refusal_count = int(self.redis.get(f"refusal:{session_id}") or 0)
            is_locked = self.is_locked_out(session_id)
            
            return {
                "enabled": True,
                "refusal_count": refusal_count,
                "is_locked_out": is_locked,
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"enabled": True, "error": str(e)}
