"""
Redis Client for Layer 3 (STWM)
Handles monotonic counters and sliding window rate limiting.

Layer 3 (Redis): "Did this user hit us 50 times in 1 second?"
Layer 4 (Faiss): "Did this user say something similar 3 months ago?"

These are DISTINCT memory systems.
"""

import redis
import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result from rate limit check."""
    is_allowed: bool
    current_count: int
    limit: int
    window_ms: int
    retry_after_ms: Optional[int] = None


class RedisSTWM:
    """
    Layer 3: Short-Term Working Memory via Redis.
    
    Handles:
    - Monotonic counters (replay attack detection)
    - Sliding window rate limiting (volumetric abuse)
    - Inter-arrival time analysis
    
    Uses Lua scripts for atomic operations.
    """
    
    # Sliding window Lua script (atomic cleanup + count + add)
    SLIDING_WINDOW_SCRIPT = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local member = ARGV[4]
        
        -- Remove elements older than window
        redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
        
        -- Count current elements
        local count = redis.call('ZCARD', key)
        
        -- Decision
        if count < limit then
            redis.call('ZADD', key, now, member)
            redis.call('PEXPIRE', key, window)
            return {1, count + 1}  -- allowed, new count
        else
            return {0, count}  -- blocked, current count
        end
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        use_mock: bool = False
    ):
        """
        Initialize Redis client.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            use_mock: Use in-memory mock instead of real Redis
        """
        self.use_mock = use_mock
        
        if use_mock:
            logger.warning("Using MOCK Redis - not suitable for production!")
            self._mock_store = {}  # Simple dict mock
            self._client = None
        else:
            try:
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Test connection
                self._client.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                logger.warning("Falling back to MOCK mode")
                self.use_mock = True
                self._mock_store = {}
                self._client = None
        
        # Register Lua script
        if not self.use_mock and self._client:
            try:
                self._sliding_window_sha = self._client.script_load(self.SLIDING_WINDOW_SCRIPT)
                logger.info("Loaded sliding window Lua script")
            except Exception as e:
                logger.error(f"Failed to load Lua script: {e}")
                self._sliding_window_sha = None
    
    def increment_counter(self, session_id: str, counter_name: str) -> int:
        """
        Increment a monotonic counter.
        
        Used for replay attack detection - sequence numbers should be monotonic.
        
        Returns:
            Current counter value
        """
        key = f"counter:{session_id}:{counter_name}"
        
        if self.use_mock:
            current = self._mock_store.get(key, 0)
            self._mock_store[key] = current + 1
            return current + 1
        
        return self._client.incr(key)
    
    def get_counter(self, session_id: str, counter_name: str) -> int:
        """Get current counter value."""
        key = f"counter:{session_id}:{counter_name}"
        
        if self.use_mock:
            return self._mock_store.get(key, 0)
        
        value = self._client.get(key)
        return int(value) if value else 0
    
    def check_rate_limit(
        self,
        session_id: str,
        limit_type: str,
        window_ms: int,
        max_requests: int
    ) -> RateLimitResult:
        """
        Check sliding window rate limit.
        
        Args:
            session_id: Session identifier
            limit_type: Type of limit (e.g., "api_calls", "prompts")
            window_ms: Window size in milliseconds
            max_requests: Maximum requests allowed in window
            
        Returns:
            RateLimitResult indicating if allowed
        """
        key = f"ratelimit:{session_id}:{limit_type}"
        now = int(time.time() * 1000)  # Milliseconds
        member = f"{now}_{id(self)}"  # Unique member
        
        if self.use_mock:
            # Simple mock implementation (not accurate sliding window)
            if key not in self._mock_store:
                self._mock_store[key] = []
            
            # Clean old entries
            self._mock_store[key] = [
                (ts, m) for ts, m in self._mock_store[key]
                if ts > now - window_ms
            ]
            
            count = len(self._mock_store[key])
            
            if count < max_requests:
                self._mock_store[key].append((now, member))
                return RateLimitResult(
                    is_allowed=True,
                    current_count=count + 1,
                    limit=max_requests,
                    window_ms=window_ms
                )
            else:
                # Calculate retry after
                oldest = min(ts for ts, _ in self._mock_store[key])
                retry_after = (oldest + window_ms) - now
                return RateLimitResult(
                    is_allowed=False,
                    current_count=count,
                    limit=max_requests,
                    window_ms=window_ms,
                    retry_after_ms=retry_after
                )
        
        # Real Redis with Lua script
        try:
            if self._sliding_window_sha:
                # Use preloaded script
                result = self._client.evalsha(
                    self._sliding_window_sha,
                    1,  # Number of keys
                    key,
                    window_ms,
                    max_requests,
                    now,
                    member
                )
            else:
                # Fallback to eval (slower)
                result = self._client.eval(
                    self.SLIDING_WINDOW_SCRIPT,
                    1,
                    key,
                    window_ms,
                    max_requests,
                    now,
                    member
                )
            
            is_allowed = bool(result[0])
            current_count = int(result[1])
            
            retry_after = None
            if not is_allowed:
                # Get oldest timestamp in window
                oldest = self._client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_ts = int(oldest[0][1])
                    retry_after = (oldest_ts + window_ms) - now
            
            return RateLimitResult(
                is_allowed=is_allowed,
                current_count=current_count,
                limit=max_requests,
                window_ms=window_ms,
                retry_after_ms=retry_after
            )
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail-open for availability
            return RateLimitResult(
                is_allowed=True,
                current_count=0,
                limit=max_requests,
                window_ms=window_ms
            )
    
    def record_timestamp(self, session_id: str, event_type: str) -> float:
        """
        Record event timestamp for inter-arrival analysis.
        
        Returns:
            Time delta from previous event (seconds)
        """
        key = f"timestamp:{session_id}:{event_type}"
        now = time.time()
        
        if self.use_mock:
            prev = self._mock_store.get(key, now)
            self._mock_store[key] = now
            return now - prev
        
        # Get previous timestamp
        prev_str = self._client.get(key)
        prev = float(prev_str) if prev_str else now
        
        # Set new timestamp
        self._client.set(key, str(now), ex=3600)  # 1 hour TTL
        
        return now - prev
    
    def reset_session(self, session_id: str):
        """Reset all keys for a session."""
        if self.use_mock:
            keys_to_delete = [k for k in self._mock_store.keys() if session_id in k]
            for k in keys_to_delete:
                del self._mock_store[k]
        else:
            pattern = f"*:{session_id}:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
    
    def health_check(self) -> bool:
        """Check if Redis is healthy."""
        if self.use_mock:
            return True
        
        try:
            return self._client.ping()
        except Exception:
            return False


# Quick test
if __name__ == "__main__":
    import time
    
    # Test with mock
    redis_client = RedisSTWM(use_mock=True)
    session = "test_session"
    
    print("Testing Redis Layer 3 (Mock Mode)\n" + "="*60)
    
    # Test 1: Monotonic counters
    print("\n1. Monotonic Counter Test:")
    for i in range(5):
        count = redis_client.increment_counter(session, "requests")
        print(f"   Request {i+1}: Counter = {count}")
    
    # Test 2: Rate limiting (10 requests in 1 second)
    print("\n2. Rate Limit Test (10 req/sec):")
    for i in range(12):
        result = redis_client.check_rate_limit(session, "api", 1000, 10)
        status = "✅ ALLOWED" if result.is_allowed else "❌ BLOCKED"
        print(f"   Request {i+1}: {status} ({result.current_count}/{result.limit})")
        time.sleep(0.05)
    
    # Test 3: Inter-arrival times
    print("\n3. Inter-Arrival Time Test:")
    for i in range(3):
        delta = redis_client.record_timestamp(session, "prompt")
        print(f"   Event {i+1}: Δt = {delta:.3f}s")
        time.sleep(0.1)
    
    print(f"\n✅ Mock mode working (Health: {redis_client.health_check()})")
