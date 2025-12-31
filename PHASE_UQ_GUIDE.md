# Phase 2: Uncertainty Quantification - User Guide

## Overview

Phase UQ adds **conformal prediction** to ALRC v5.0, providing prediction SETS instead of point predictions:

```
{Safe}        → Confident safe → allow
{Unsafe}      → Confident unsafe → block  
{Safe, Unsafe}→ UNCERTAIN → escalate/steer
```

**Coverage Guarantee**: True label is in prediction set ≥95% of the time.

---

## Installation

Phase UQ is already installed as part of `src/safety/alrc/phase_uq/`.

No additional dependencies needed (uses NumPy).

---

## Quick Start

### 1. Generate Calibration Dataset

Create `datasets/calibration/calibration.jsonl` with format:

```jsonl
{"text": "How do I make a bomb?", "label": "unsafe", "source": "..."}
{"text": "Explain encryption", "label": "safe", "source": "..."}
```

**Minimum**: 500 samples (250 safe, 250 unsafe)  
**Recommended**: 1500+ samples

### 2. Run Calibration

```bash
python scripts/calibrate_uq.py \
    --dataset datasets/calibration/sample.jsonl \
    --alpha 0.05 \
    --output models/conformal.json
```

Output:
```
Calibration complete:
  τ_safe = 0.3421
  τ_unsafe = 0.6789
  Model version: abc123def456
✅ Saved calibrated model to: models/conformal.json
```

### 3. Enable in Pipeline

```python
from src.safety.alrc.pipeline import ALRCPipeline

pipeline = ALRCPipeline(
    enable_uq=True,  # Enable Phase UQ
    uq_model_path="models/conformal.json"
)

result = pipeline.analyze("How do I hack a website?", "session_1")

print(f"Prediction set: {result.prediction_set}")
print(f"Uncertain: {result.is_uncertain}")
print(f"Action: {result.action}")
```

---

## Configuration

### Escalation Policy

```python
# Conservative (production default)
policy = EscalationPolicy(
    enable_human_review=False,
    conservative=True  # uncertain → steer (not allow)
)

# Permissive (testing only)
policy = EscalationPolicy(
    conservative=False  # uncertain → allow
)
```

### MVP Groups (Advanced)

Enable group-conditional coverage:

```python
from src.safety.alrc.phase_uq import MVPPredictor

mvp = MVPPredictor(model_version="v1")

uq_filter = UncertaintyFilter(
    predictor=mvp,
    use_mvp=True  # Enable MVP
)
```

**Groups**:
- `high_risk`: risk > 0.7
- `ambiguous`: 0.3 < risk < 0.7
- `high_escalation`: escalation > 0.6
- `long_prompt`: >200 tokens
- `code_heavy`: Contains code
- `high_drift`: long_term_drift > 0.3
- `refusal_history`: Recent refusals

---

## Monitoring

### View Logs

```bash
tail -f logs/uq.jsonl
```

Sample output:
```json
{
  "timestamp": 1704067200.0,
  "session_id": "abc123",
  "risk_score": 0.65,
  "prediction_set": ["safe", "unsafe"],
  "is_uncertain": true,
  "action": "steer"
}
```

### Compute Metrics

```python
from src.safety.alrc.phase_uq import UQMonitor

monitor = UQMonitor()
logs = monitor.load_logs("logs/uq.jsonl")
metrics = monitor.compute_metrics(logs)

print(f"Uncertainty rate: {metrics['uncertainty_rate']:.1%}")
print(f"Per-group uncertainty: {metrics['per_group_uncertainty']}")

# Check if recalibration needed
if monitor.should_recalibrate(metrics):
    print("⚠️  Recalibration recommended")
```

**Triggers**:
- Uncertainty rate > 20%
- Any group uncertainty > 30%

---

## Testing

### Run Tests

```bash
pytest phase_uq_tests.py -v
```

Expected output:
```
test_stratified_calibration PASSED
test_prediction_set PASSED
test_coverage_guarantee PASSED
test_hard_override PASSED
test_conservative_fallback PASSED
...
================== 20 passed in 2.34s ==================
```

### Validate Coverage

```python
from scripts.calibrate_uq import validate_coverage

coverage = validate_coverage(pipeline, predictor, test_samples)
print(f"Coverage: {coverage:.1%}")  # Should be ≥93%
```

---

## Recalibration

### When to Recalibrate

- Model version changed
- Uncertainty rate > 20%
- Coverage dropped (if measurable)
- New attack types discovered

### How to Recalibrate

```bash
# 1. Add new samples to calibration dataset
cat new_samples.jsonl >> datasets/calibration/calibration.jsonl

# 2. Re-run calibration
python scripts/calibrate_uq.py \
    --dataset datasets/calibration/calibration.jsonl \
    --alpha 0.05 \
    --output models/conformal.json

# 3. Restart pipeline (will load new model)
```

---

## Rollback

### Disable Phase UQ

```python
pipeline = ALRCPipeline(
    enable_uq=False  # Disable
)
```

Or via environment variable:

```bash
export ENABLE_PHASE_UQ=false
```

---

## Troubleshooting

### High Uncertainty Rate (>25%)

**Cause**: Calibration set unrepresentative

**Fix**: Add more diverse samples, especially edge cases

### Coverage Below 93%

**Cause**: Test set distribution != calibration set

**Fix**: Stratified calibration (already default), or expand calibration set

### Model Version Mismatch

**Symptom**: Warning on startup

**Cause**: Pipeline changed since calibration

**Fix**: Recalibrate with new model version

---

## Advanced: Custom Score Functions

Replace margin score with custom non-conformity:

```python
class CustomScorer:
    @staticmethod
    def custom_score(probs, label):
        # Your custom logic
        return 1.0 - probs[label]

# Use in calibration
# (Would need to modify calibration script)
```

---

## Performance

| Metric | Value |
|--------|-------|
| Latency (per query) | <1ms |
| Memory footprint | ~10KB |
| Calibration time | ~3 min (1500 samples) |

---

## References

- [Conformal Prediction](https://arxiv.org/abs/2107.07511) - Angelopoulos & Bates
- [MVP](https://arxiv.org/abs/2112.03882) - Multivalid Prediction
- Phase 2 Implementation Plan: [phase2_implementation_plan.md](file:///C:/Users/zakup/.gemini/antigravity/brain/e66f0612-5876-4362-a21b-2f7d029387cf/phase2_implementation_plan.md)
