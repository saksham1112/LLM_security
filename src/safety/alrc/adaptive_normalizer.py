"""
Layer 1: Adaptive Normalizer (Numba JIT Optimized)
Streaming Welford algorithm with EWMA for concept drift handling.

Uses Numba JIT compilation for <2ms latency.
"""

import numpy as np
from numba import jit, float64
from dataclasses import dataclass
from typing import Dict, Optional
import logging
import struct

logger = logging.getLogger(__name__)


# ============================================================================
# Numba JIT-Compiled Kernels (Pure machine code, ~200ns per call)
# ============================================================================

@jit(nopython=True, cache=True)
def update_streaming_stats(current_mean: float, current_var: float, 
                           new_value: float, alpha: float) -> tuple:
    """
    JIT-compiled EWMA Welford update kernel.
    
    Math:
        δ = x_t - μ_{t-1}
        μ_t = μ_{t-1} + α * δ
        Var_t = (1-α) * (Var_{t-1} + α * δ²)
    
    Execution time: ~200 nanoseconds
    """
    diff = new_value - current_mean
    
    # Update mean
    updated_mean = current_mean + alpha * diff
    
    # Update variance (EWMA Welford)
    updated_var = (1.0 - alpha) * (current_var + alpha * (diff ** 2))
    
    return updated_mean, updated_var


@jit(nopython=True, cache=True)
def calculate_z_score(value: float, mean: float, var: float) -> float:
    """
    JIT-compiled Z-score calculation with division-by-zero protection.
    """
    if var <= 1e-9:
        return 0.0
    std_dev = np.sqrt(var)
    return (value - mean) / std_dev


@jit(nopython=True, cache=True)
def calculate_anomaly_score(z_score: float) -> float:
    """
    Convert Z-score to anomaly probability using sigmoid.
    Higher Z → Higher anomaly.
    """
    # sigmoid((|z| - 2) * 2) - shifted to trigger at z > 2
    abs_z = abs(z_score)
    if abs_z < 1.0:
        return 0.0
    x = (abs_z - 2.0) * 2.0
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NormalizerState:
    """State for a single metric dimension."""
    mean: float = 0.0
    var: float = 0.0
    count: int = 0


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    z_score: float
    is_anomaly: bool
    risk_contribution: float
    dimension: str


# ============================================================================
# Main Normalizer Class
# ============================================================================

class AdaptiveNormalizer:
    """
    Production-grade adaptive normalizer with Numba JIT optimization.
    
    Features:
    - EWMA Welford for streaming variance
    - Numba JIT compilation (~200ns per update)
    - Float64 precision for numerical stability
    - Cold-start handling with global priors
    """
    
    # EWMA decay factor (α = 0.01 → ~100 event memory)
    DEFAULT_ALPHA = 0.01
    
    # Cold start threshold
    MIN_SAMPLES = 50
    
    # Anomaly thresholds
    ANOMALY_Z_THRESHOLD = 2.5  # Standard deviations
    
    def __init__(self, alpha: float = DEFAULT_ALPHA):
        """Initialize with decay factor."""
        self.alpha = alpha
        
        # Per-session, per-dimension state
        # Format: {session_id: {dimension: NormalizerState}}
        self._states: Dict[str, Dict[str, NormalizerState]] = {}
        
        # Global priors (for cold start)
        self._global_priors = {
            "message_length": NormalizerState(mean=50.0, var=2500.0, count=100),
            "risk_score": NormalizerState(mean=0.2, var=0.04, count=100),
        }
        
        # Trigger JIT compilation on first import
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Pre-compile Numba functions to avoid first-call latency."""
        try:
            # Warmup calls
            _ = update_streaming_stats(0.0, 1.0, 0.5, 0.01)
            _ = calculate_z_score(0.5, 0.0, 1.0)
            _ = calculate_anomaly_score(2.0)
            logger.debug("Numba JIT warmup complete")
        except Exception as e:
            logger.warning(f"Numba warmup failed: {e}")
    
    def _get_state(self, session_id: str, dimension: str) -> NormalizerState:
        """Get or create state for session/dimension."""
        if session_id not in self._states:
            self._states[session_id] = {}
        
        if dimension not in self._states[session_id]:
            # Initialize with global prior if available
            if dimension in self._global_priors:
                prior = self._global_priors[dimension]
                self._states[session_id][dimension] = NormalizerState(
                    mean=prior.mean,
                    var=prior.var,
                    count=0  # Start fresh but with prior estimates
                )
            else:
                self._states[session_id][dimension] = NormalizerState()
        
        return self._states[session_id][dimension]
    
    def update(
        self, 
        session_id: str, 
        dimension: str, 
        value: float
    ) -> AnomalyResult:
        """
        Update streaming statistics and detect anomalies.
        
        Args:
            session_id: Session identifier
            dimension: Metric dimension (e.g., "message_length")
            value: New observation
            
        Returns:
            AnomalyResult with Z-score and risk contribution
        """
        state = self._get_state(session_id, dimension)
        
        # Calculate Z-score BEFORE update (to detect anomaly in new value)
        z_score = calculate_z_score(value, state.mean, state.var)
        
        # Update streaming statistics (JIT compiled)
        state.mean, state.var = update_streaming_stats(
            state.mean, state.var, value, self.alpha
        )
        state.count += 1
        
        # Determine if anomaly
        is_anomaly = abs(z_score) > self.ANOMALY_Z_THRESHOLD and state.count > self.MIN_SAMPLES
        
        # Calculate risk contribution
        risk_contribution = calculate_anomaly_score(z_score) if state.count > self.MIN_SAMPLES else 0.0
        
        return AnomalyResult(
            z_score=z_score,
            is_anomaly=is_anomaly,
            risk_contribution=min(1.0, risk_contribution),
            dimension=dimension
        )
    
    def get_session_stats(self, session_id: str) -> Dict[str, NormalizerState]:
        """Get all stats for a session."""
        return self._states.get(session_id, {})
    
    def reset_session(self, session_id: str):
        """Reset all state for a session."""
        if session_id in self._states:
            del self._states[session_id]
    
    def serialize_state(self, session_id: str, dimension: str) -> bytes:
        """Serialize state to binary for Redis storage."""
        state = self._get_state(session_id, dimension)
        # Pack as: mean (float64), var (float64), count (uint64)
        return struct.pack('>ddQ', state.mean, state.var, state.count)
    
    def deserialize_state(self, data: bytes, session_id: str, dimension: str):
        """Deserialize state from Redis."""
        mean, var, count = struct.unpack('>ddQ', data)
        if session_id not in self._states:
            self._states[session_id] = {}
        self._states[session_id][dimension] = NormalizerState(mean, var, count)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    import time
    
    normalizer = AdaptiveNormalizer()
    session = "test_session"
    
    # Simulate normal traffic
    print("Simulating normal message lengths:")
    normal_lengths = np.random.normal(50, 15, 100)  # Mean=50, Std=15
    
    start = time.time()
    for length in normal_lengths:
        result = normalizer.update(session, "message_length", length)
    normal_time = (time.time() - start) * 1000
    print(f"  100 updates: {normal_time:.2f}ms ({normal_time/100:.3f}ms per update)")
    
    # Now inject anomaly
    print("\nDetecting anomalies:")
    anomalies = [200, 5, 300, 2, 500]
    for length in anomalies:
        result = normalizer.update(session, "message_length", length)
        status = "⚠️ ANOMALY" if result.is_anomaly else "✓ Normal"
        print(f"  {status}: length={length}, z={result.z_score:.2f}, risk={result.risk_contribution:.2f}")
    
    # Benchmark
    print("\nBenchmark (1000 updates):")
    start = time.time()
    for _ in range(1000):
        normalizer.update(session, "bench", np.random.random())
    bench_time = (time.time() - start) * 1000
    print(f"  Total: {bench_time:.2f}ms, Per-update: {bench_time/1000:.3f}ms")
