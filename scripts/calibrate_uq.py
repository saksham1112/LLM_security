"""Calibration script for Phase UQ conformal predictor.

Usage:
    python scripts/calibrate_uq.py \\
        --dataset datasets/calibration.jsonl \\
        --alpha 0.05 \\
        --output models/conformal.json
"""

import argparse
import json
import asyncio
import numpy as np
import hashlib
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety.alrc.pipeline import ALRCPipeline
from src.safety.alrc.phase_uq import SplitConformalPredictor


def load_calibration_dataset(path: str):
    """
    Load calibration dataset from JSONL.
    
    Format:
    {"text": "...", "label": "safe"|"unsafe", "source": "..."}
    """
    samples = []
    
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)
    
    print(f"Loaded {len(samples)} calibration samples")
    
    # Check label distribution
    safe_count = sum(1 for s in samples if s['label'] == 'safe')
    unsafe_count = sum(1 for s in samples if s['label'] == 'unsafe')
    
    print(f"  Safe: {safe_count} ({safe_count/len(samples):.1%})")
    print(f"  Unsafe: {unsafe_count} ({unsafe_count/len(samples):.1%})")
    
    return samples


def compute_model_hash(pipeline):
    """Compute hash of model components for version binding."""
    components = []
    
    # Hash semantic model
    if hasattr(pipeline.semantic, 'model_name'):
        components.append(pipeline.semantic.model_name)
    
    # Hash trajectory weights
    if pipeline.trajectory:
        weights = [
            pipeline.trajectory.WEIGHT_PROXIMITY,
            pipeline.trajectory.WEIGHT_INTEGRAL,
            pipeline.trajectory.WEIGHT_INTENT,
            pipeline.trajectory.WEIGHT_BACKTRACK,
            pipeline.trajectory.WEIGHT_LONG_TERM,
        ]
        components.append(str(weights))
    
    combined = "|".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


async def calibrate_scp(pipeline, samples, alpha=0.05):
    """
    Calibrate SCP with stratified calibration.
    
    Returns:
        SplitConformalPredictor (calibrated)
    """
    print(f"\nCalibrating SCP with alpha={alpha}...")
    
    # Separate by label
    safe_samples = [s for s in samples if s['label'] == 'safe']
    unsafe_samples = [s for s in samples if s['label'] == 'unsafe']
    
    print(f"  Safe samples: {len(safe_samples)}")
    print(f"  Unsafe samples: {len(unsafe_samples)}")
    
    # Compute non-conformity scores per label
    safe_scores = []
    unsafe_scores = []
    
    print("\nComputing scores...")
    
    for sample in safe_samples:
        result = await pipeline.analyze(sample['text'], session_id="calibration")
        risk = result.risk_score
        
        # Non-conformity for true label "safe"
        score = risk  # s(x, safe) = 1 - p(safe) = 1 - (1 - risk) = risk
        safe_scores.append(score)
    
    for sample in unsafe_samples:
        result = await pipeline.analyze(sample['text'], session_id="calibration")
        risk = result.risk_score
        
        # Non-conformity for true label "unsafe"
        score = 1.0 - risk  # s(x, unsafe) = 1 - p(unsafe) = 1 - risk
        unsafe_scores.append(score)
    
    # Create and calibrate predictor
    model_version = compute_model_hash(pipeline)
    predictor = SplitConformalPredictor(alpha=alpha, model_version=model_version)
    
    thresholds = predictor.calibrate_stratified(
        safe_scores=np.array(safe_scores),
        unsafe_scores=np.array(unsafe_scores)
    )
    
    print(f"\nCalibration complete:")
    print(f"  τ_safe = {thresholds['safe']:.4f}")
    print(f"  τ_unsafe = {thresholds['unsafe']:.4f}")
    print(f"  Model version: {model_version}")
    
    return predictor


async def validate_coverage(pipeline, predictor, test_samples):
    """
    Validate coverage on held-out test set.
    
    Returns:
        coverage_rate: float
    """
    print(f"\nValidating coverage on {len(test_samples)} test samples...")
    
    covered = 0
    total = 0
    
    for sample in test_samples:
        result = await pipeline.analyze(sample['text'], session_id="test")
        risk = result.risk_score
        
        probs = {"safe": 1 - risk, "unsafe": risk}
        conf_result = predictor.predict_set(probs)
        
        if sample['label'] in conf_result.prediction_set:
            covered += 1
        
        total += 1
    
    coverage = covered / total if total > 0 else 0.0
    
    print(f"Coverage: {coverage:.1%} (target: ≥{1 - predictor.alpha:.1%})")
    
    return coverage


async def main():
    parser = argparse.ArgumentParser(description="Calibrate Phase UQ")
    parser.add_argument('--dataset', required=True, help="Path to calibration JSONL")
    parser.add_argument('--alpha', type=float, default=0.05, help="Error rate (0.05 = 95% coverage)")
    parser.add_argument('--output', required=True, help="Output path for calibrated model")
    parser.add_argument('--test-split', type=float, default=0.2, help="Test set fraction")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase UQ Calibration")
    print("=" * 60)
    
    # Load dataset
    samples = load_calibration_dataset(args.dataset)
    
    # Train/test split
    np.random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.test_split))
    
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    print(f"\nSplit: {len(train_samples)} train, {len(test_samples)} test")
    
    # Initialize pipeline
    print("\nInitializing ALRC pipeline...")
    pipeline = ALRCPipeline(
        enable_trajectory=True,
        enable_long_term=False  # UQ doesn't need Phase L for calibration
    )
    
    # Calibrate
    predictor = await calibrate_scp(pipeline, train_samples, alpha=args.alpha)
    
    # Validate
    if test_samples:
        coverage = await validate_coverage(pipeline, predictor, test_samples)
        
        if coverage < (1 - args.alpha) - 0.02:  # Allow 2% slack
            print(f"\n⚠️  WARNING: Coverage below target!")
        else:
            print(f"\n✅ Coverage OK")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    predictor.save(str(output_path))
    
    print(f"\n✅ Saved calibrated model to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
