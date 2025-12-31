"""
Latency benchmarks for Phase L.

Measures:
- Redis p99 latency under concurrent load
- Phase L overhead per call
- Prototype update latency

Run with: python latency_benchmark.py
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
import statistics

from src.safety.alrc.phase_l.integration import LongTermMemory


def benchmark_redis_latency(num_sessions: int = 100, turns_per_session: int = 10):
    """
    Benchmark Redis latency under concurrent load.
    
    Args:
        num_sessions: Number of concurrent sessions
        turns_per_session: Turns per session
    """
    print(f"\n=== Redis Latency Benchmark ===")
    print(f"Sessions: {num_sessions}, Turns/session: {turns_per_session}")
    
    # Setup
    hazards = [np.ones(384, dtype=np.float32)]
    ltm = LongTermMemory(
        redis_url="redis://localhost:6379",
        hazard_centroids=hazards,
        enabled=True
    )
    
    def session_workload(session_id: int) -> List[float]:
        """Simulate a session with multiple turns."""
        latencies = []
        
        for turn in range(turns_per_session):
            emb = np.random.rand(384).astype(np.float32)
            intent = {'educational': 0.5, 'operational': 0.3, 'instructional': 0.2, 'malicious': 0.0}
            
            start = time.perf_counter()
            ltm.update(f"session_{session_id}", emb, 0.5, intent)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        return latencies
    
    # Run concurrent sessions
    all_latencies = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(session_workload, i) for i in range(num_sessions)]
        for future in futures:
            all_latencies.extend(future.result())
    
    # Calculate statistics
    p50 = statistics.median(all_latencies)
    p95 = statistics.quantiles(all_latencies, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(all_latencies, n=100)[98]  # 99th percentile
    mean = statistics.mean(all_latencies)
    
    print(f"\nResults:")
    print(f"  Mean:   {mean:.2f}ms")
    print(f"  P50:    {p50:.2f}ms")
    print(f"  P95:    {p95:.2f}ms")
    print(f"  P99:    {p99:.2f}ms")
    
    # Check if we meet target (<3ms p99)
    if p99 < 3.0:
        print(f"  ✅ PASS: P99 {p99:.2f}ms < 3ms target")
    else:
        print(f"  ❌ FAIL: P99 {p99:.2f}ms >= 3ms target")
    
    return {'mean': mean, 'p50': p50, 'p95': p95, 'p99': p99}


def benchmark_prototype_update():
    """Benchmark UserPrototype update latency."""
    from src.safety.alrc.phase_l.prototype import UserPrototype
    
    print(f"\n=== Prototype Update Benchmark ===")
    
    prototype = UserPrototype(embedding_dim=384, alpha=0.05)
    latencies = []
    
    # Warm-up
    for _ in range(100):
        emb = np.random.rand(384).astype(np.float32)
        prototype.update(emb)
    
    # Measure
    for _ in range(1000):
        emb = np.random.rand(384).astype(np.float32)
        start = time.perf_counter()
        prototype.update(emb)
        latency_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(latency_us)
    
    mean = statistics.mean(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]
    
    print(f"  Mean:   {mean:.1f}µs")
    print(f"  P99:    {p99:.1f}µs")
    print(f"  ✅ (NumPy operations - very fast)")
    
    return {'mean_us': mean, 'p99_us': p99}


def benchmark_drift_computation():
    """Benchmark DriftTracker computation latency."""
    from src.safety.alrc.phase_l.drift import DriftTracker
    
    print(f"\n=== Drift Computation Benchmark ===")
    
    hazards = [np.random.rand(384).astype(np.float32) for _ in range(5)]
    tracker = DriftTracker(hazard_centroids=hazards, decay=0.9)
    latencies = []
    
    # Measure
    for _ in range(1000):
        prototype = np.random.rand(384).astype(np.float32)
        start = time.perf_counter()
        tracker.compute_drift(prototype)
        latency_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(latency_us)
    
    mean = statistics.mean(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]
    
    print(f"  Mean:   {mean:.1f}µs")
    print(f"  P99:    {p99:.1f}µs")
    print(f"  ✅ (Cosine similarity - very fast)")
    
    return {'mean_us': mean, 'p99_us': p99}


def benchmark_end_to_end():
    """Benchmark full Phase L overhead."""
    print(f"\n=== End-to-End Phase L Overhead ===")
    
    hazards = [np.ones(384, dtype=np.float32)]
    ltm = LongTermMemory(
        redis_url="redis://localhost:6379",
        hazard_centroids=hazards,
        enabled=True
    )
    
    latencies = []
    
    # Warm-up
    for i in range(10):
        emb = np.random.rand(384).astype(np.float32)
        ltm.update(f"warmup_{i}", emb, 0.5, {})
    
    # Measure
    for i in range(100):
        emb = np.random.rand(384).astype(np.float32)
        intent = {'educational': 0.5, 'operational': 0.3, 'instructional': 0.2, 'malicious': 0.0}
        
        start = time.perf_counter()
        drift = ltm.update(f"bench_{i}", emb, 0.5, intent)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    mean = statistics.mean(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]
    
    print(f"  Mean:   {mean:.2f}ms")
    print(f"  P99:    {p99:.2f}ms")
    
    if p99 < 3.0:
        print(f"  ✅ PASS: Meets <3ms target")
    else:
        print(f"  ⚠️  WARNING: Exceeds 3ms target")
    
    return {'mean': mean, 'p99': p99}


if __name__ == "__main__":
    print("=" * 60)
    print("Phase L Latency Benchmarks")
    print("=" * 60)
    
    # Run benchmarks
    redis_stats = benchmark_redis_latency(num_sessions=100, turns_per_session=10)
    prototype_stats = benchmark_prototype_update()
    drift_stats = benchmark_drift_computation()
    e2e_stats = benchmark_end_to_end()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Redis P99:        {redis_stats['p99']:.2f}ms")
    print(f"Prototype P99:    {prototype_stats['p99_us']:.1f}µs")
    print(f"Drift P99:        {drift_stats['p99_us']:.1f}µs")
    print(f"End-to-End P99:   {e2e_stats['p99']:.2f}ms")
    print("\nTarget: <3ms P99 for Phase L")
    
    if e2e_stats['p99'] < 3.0:
        print("✅ PASS: Phase L meets latency target")
    else:
        print("❌ FAIL: Phase L exceeds latency target")
