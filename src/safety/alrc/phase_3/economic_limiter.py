"""Economic Rate Limiter: Cost-based attack prevention."""

from typing import Dict
import logging

try:
    from redis import Redis
except ImportError:
    Redis = None

logger = logging.getLogger(__name__)


class EconomicRateLimiter:
    """
    Dynamic rate limiting based on cost escalation.
    
    Principle: Make attacks expensive without hurting legitimate users.
    """
    
    # Base costs (abstract units)
    COST_PER_QUERY = 1
    COST_PER_REFUSAL = 5       # Probing is expensive
    COST_PER_UNCERTAIN = 3     # UQ escalation
    COST_PER_LOCKOUT = 20      # Lockout is very expensive
    
    # Revenue thresholds
    MAX_COST_PER_USER_HOUR = 50
    MAX_COST_PER_IP_HOUR = 200
    
    def __init__(self, redis_url: str = "redis://localhost:6379", enabled: bool = True):
        self.enabled = enabled and Redis is not None
        
        if self.enabled:
            try:
                self.redis = Redis.from_url(redis_url, decode_responses=True)
                self.redis.ping()
                logger.info(f"EconomicRateLimiter initialized with Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Economic limiter disabled.")
                self.enabled = False
                self.redis = None
        else:
            self.redis = None
            logger.info("EconomicRateLimiter disabled (Redis not available)")
    
    def track_cost(self, session_id: str, event_type: str) -> float:
        """
        Track cumulative cost for session.
        
        Args:
            event_type: "query", "refusal", "uncertain", "lockout"
        
        Returns:
            Current total cost
        """
        if not self.enabled:
            return 0.0
        
        cost_map = {
            "query": self.COST_PER_QUERY,
            "refusal": self.COST_PER_REFUSAL,
            "uncertain": self.COST_PER_UNCERTAIN,
            "lockout": self.COST_PER_LOCKOUT,
        }
        
        cost = cost_map.get(event_type, 0)
        
        try:
            key = f"cost:{session_id}"
            total = self.redis.incrbyfloat(key, cost)
            self.redis.expire(key, 3600)  # 1 hour window
            
            return float(total)
        except Exception as e:
            logger.error(f"Cost tracking failed: {e}")
            return 0.0
    
    def should_rate_limit(self, session_id: str) -> bool:
        """Check if session has exceeded cost threshold."""
        if not self.enabled:
            return False
        
        try:
            key = f"cost:{session_id}"
            total = float(self.redis.get(key) or 0)
            
            exceeded = total >= self.MAX_COST_PER_USER_HOUR
            
            if exceeded:
                logger.warning(
                    f"Economic rate limit exceeded: session={session_id[:8]}, "
                    f"cost={total:.1f}"
                )
            
            return exceeded
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    def get_wait_time(self, session_id: str) -> int:
        """
        Calculate exponential backoff.
        
        Returns:
            Seconds to wait before retry
        """
        if not self.enabled:
            return 0
        
        try:
            key = f"cost:{session_id}"
            total = float(self.redis.get(key) or 0)
            
            if total < self.MAX_COST_PER_USER_HOUR:
                return 0
            
            # Exponential backoff
            excess = total - self.MAX_COST_PER_USER_HOUR
            wait = min(300, int(2 ** (excess / 10)))  # Max 5 min
            
            return wait
        except Exception as e:
            logger.error(f"Wait time calculation failed: {e}")
            return 0
    
    def get_stats(self, session_id: str) -> Dict:
        """Get cost statistics for session."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            key = f"cost:{session_id}"
            total = float(self.redis.get(key) or 0)
            
            return {
                "enabled": True,
                "total_cost": total,
                "threshold": self.MAX_COST_PER_USER_HOUR,
                "should_limit": total >= self.MAX_COST_PER_USER_HOUR,
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"enabled": True, "error": str(e)}
