# Phase L: How to Run

## Prerequisites

```bash
# Install dependencies
pip install redis numpy pytest

# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:latest
```

## Run Tests

```bash
# Run all tests
pytest phase_l_tests.py -v

# Run specific test class
pytest phase_l_tests.py::TestSessionMemory -v

# Run with coverage
pytest phase_l_tests.py --cov=src.safety.alrc.phase_l --cov-report=html
```

## Run Benchmarks

```bash
# Run latency benchmarks
python latency_benchmark.py

# Expected output:
# === Redis Latency Benchmark ===
# Sessions: 100, Turns/session: 10
# 
# Results:
#   Mean:   1.23ms
#   P50:    1.15ms
#   P95:    2.10ms
#   P99:    2.45ms
#   ✅ PASS: P99 2.45ms < 3ms target
```

## Interpret Results

### Test Results

- **All tests passing**: Phase L is working correctly
- **TTL test fails**: Check Redis connection
- **Prototype test fails**: Check NumPy installation

### Benchmark Results

| Metric | Target | Good | Warning | Bad |
|--------|--------|------|---------|-----|
| Redis P99  | <3ms | <2ms | 2-3ms | >3ms |
| E2E P99    | <3ms | <2.5ms | 2.5-3ms | >3ms |
| Prototype  | N/A | <100µs | <500µs | >1ms |

**If P99 > 3ms:**
- Check Redis is local (not network)
- Reduce `MAX_TURNS` to lower memory
- Use Redis pipelining (future optimization)
- Check concurrent load

## Integration Test

```bash
# Test Phase L integration with pipeline
python -c "
from src.safety.alrc.phase_l import LongTermMemory
import numpy as np

ltm = LongTermMemory(enabled=True)
emb = np.random.rand(384).astype(np.float32)
drift = ltm.update('test', emb, 0.5, {})
print(f'Drift: {drift:.4f}')
"
```

## Monitoring in Production

```python
# Add to pipeline monitoring
stats = pipeline.phase_l.get_stats()

# Track metrics:
# - active_sessions: Should match concurrent users
# - cached_prototypes: Should grow up to active sessions
# - If cached > active: Memory leak (investigate)
```

## Troubleshooting

### Redis Connection Error

```
Error: Redis connection failed: [Errno 111] Connection refused
```

**Fix**: Start Redis server

```bash
redis-server
# Or check if running: redis-cli ping
```

### Memory Pressure

```
Warning: Redis memory usage high
```

**Fix**: Tune TTL and MAX_TURNS

```python
SessionMemory.TTL_SECONDS = 900  # 15 min instead of 30
SessionMemory.MAX_TURNS = 50     # 50 instead of 100
```

### High Latency

```
P99: 5.2ms (exceeds 3ms target)
```

**Fix**:
1. Check Redis is on localhost
2. Use pipelining for batch operations
3. Reduce concurrent connections
4. Consider Redis Cluster for scale
