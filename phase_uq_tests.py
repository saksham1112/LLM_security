"""Tests for Phase UQ: Uncertainty Quantification.

Run:
    pytest phase_uq_tests.py -v
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.safety.alrc.phase_uq import (
    SplitConformalPredictor,
    MVPPredictor,
    get_groups,
    EscalationPolicy,
    UncertaintyFilter,
    UQLogger,
    UQMonitor,
)


class TestConformalPredictor:
    """Test conformal prediction core."""
    
    def test_stratified_calibration(self):
        """Test per-label calibration."""
        predictor = SplitConformalPredictor(alpha=0.05)
        
        # Generate calibration scores
        safe_scores = np.random.uniform(0.0, 0.4, 100)   # Safe samples
        unsafe_scores = np.random.uniform(0.6, 1.0, 100) # Unsafe samples
        
        thresholds = predictor.calibrate_stratified(safe_scores, unsafe_scores)
        
        assert predictor.calibrated
        assert 'safe' in thresholds
        assert 'unsafe' in thresholds
        assert thresholds['safe'] < 0.5  # Should be low for safe
        assert thresholds['unsafe'] > 0.5  # Should be high for unsafe
    
    def test_prediction_set(self):
        """Test prediction set construction."""
        predictor = SplitConformalPredictor(alpha=0.05)
        
        safe_scores = np.array([0.1, 0.2, 0.3])
        unsafe_scores = np.array([0.7, 0.8, 0.9])
        predictor.calibrate_stratified(safe_scores, unsafe_scores)
        
        # Test confident safe
        result = predictor.predict_set({"safe": 0.9, "unsafe": 0.1})
        assert "safe" in result.prediction_set
        assert len(result.prediction_set) == 1
        assert notresult.is_uncertain
        
        # Test confident unsafe
        result = predictor.predict_set({"safe": 0.1, "unsafe": 0.9})
        assert "unsafe" in result.prediction_set
        assert len(result.prediction_set) == 1
        assert not result.is_uncertain
        
        # Test uncertain (both should be included)
        result = predictor.predict_set({"safe": 0.5, "unsafe": 0.5})
        assert len(result.prediction_set) >= 1  # Should include at least one
    
    def test_coverage_guarantee(self):
        """Test that calibration achieves target coverage."""
        predictor = SplitConformalPredictor(alpha=0.05)
        
        # Calibrate
        n = 200
        safe_scores = np.random.uniform(0.0, 0.5, n)
        unsafe_scores = np.random.uniform(0.5, 1.0, n)
        predictor.calibrate_stratified(safe_scores, unsafe_scores)
        
        # Test on held-out set
        covered = 0
        total = 100
        
        for _ in range(total):
            true_label = "safe" if np.random.rand() < 0.5 else "unsafe"
            risk = np.random.uniform(0.0, 0.5) if true_label == "safe" else np.random.uniform(0.5, 1.0)
            
            probs = {"safe": 1 - risk, "unsafe": risk}
            result = predictor.predict_set(probs)
            
            if true_label in result.prediction_set:
                covered += 1
        
        coverage = covered / total
        # Should be >= 1 - alpha (95%), allow some slack for randomness
        assert coverage >= 0.90, f"Coverage {coverage:.1%} < 90%"
    
    def test_save_load(self):
        """Test serialization."""
        predictor = SplitConformalPredictor(alpha=0.05, model_version="test_v1")
        
        safe_scores = np.array([0.1, 0.2, 0.3])
        unsafe_scores = np.array([0.7, 0.8, 0.9])
        predictor.calibrate_stratified(safe_scores, unsafe_scores)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            path = f.name
        
        try:
            predictor.save(path)
            loaded = SplitConformalPredictor.load(path)
            
            assert loaded.alpha == predictor.alpha
            assert loaded.tau_safe == predictor.tau_safe
            assert loaded.tau_unsafe == predictor.tau_unsafe
            assert loaded.model_version == predictor.model_version
        finally:
            Path(path).unlink()


class TestMVPGroups:
    """Test MVP group detection."""
    
    def test_high_risk_group(self):
        """Test high risk group detection."""
        groups = get_groups(
            query="test query",
            risk_score=0.8,
            escalation_score=0.0,
            long_term_drift=0.0,
            session_refusals=0
        )
        
        assert "high_risk" in groups
        assert "ambiguous" not in groups
    
    def test_ambiguous_group(self):
        """Test ambiguous group detection."""
        groups = get_groups(
            query="test query",
            risk_score=0.5,
            escalation_score=0.0
        )
        
        assert "ambiguous" in groups
        assert "high_risk" not in groups
    
    def test_long_prompt_group(self):
        """Test long prompt detection."""
        long_query = " ".join(["word"] * 250)
        groups = get_groups(
            query=long_query,
            risk_score=0.5,
            escalation_score=0.0
        )
        
        assert "long_prompt" in groups
    
    def test_code_detection(self):
        """Test code block detection."""
        code_query = "```python\ndef test():\n    pass\n```"
        groups = get_groups(
            query=code_query,
            risk_score=0.3,
            escalation_score=0.0
        )
        
        assert "code_heavy" in groups
    
    def test_drift_group(self):
        """Test drift group detection."""
        groups = get_groups(
            query="test",
            risk_score=0.3,
            escalation_score=0.0,
            long_term_drift=0.5
        )
        
        assert "high_drift" in groups


class TestMVPPredictor:
    """Test MVP predictor."""
    
    def test_group_threshold_selection(self):
        """Test that MVP uses minimum (strictest) threshold."""
        mvp = MVPPredictor(model_version="test")
        
        # High risk group should use strict threshold
        result = mvp.predict_set(
            probs={"safe": 0.5, "unsafe": 0.5},
            groups={"high_risk"}
        )
        
        # With strict threshold (0.85), should be uncertain
        assert result.is_uncertain or len(result.prediction_set) == 2
    
    def test_multiple_groups(self):
        """Test that strictest group wins."""
        mvp = MVPPredictor()
        
        # high_risk (0.85) + ambiguous (0.95)
        # Should use min = 0.85 (stricter)
        result = mvp.predict_set(
            probs={"safe": 0.6, "unsafe": 0.4},
            groups={"high_risk", "ambiguous"}
        )
        
        # Verify strict threshold was used
        assert result.tau_used == 0.85


class TestEscalationPolicy:
    """Test escalation policy."""
    
    def test_confident_predictions(self):
        """Test confident predictions."""
        policy = EscalationPolicy(conservative=True)
        
        # Confident safe
        action = policy.decide({"safe"}, "benign", 0.0)
        assert action == "allow"
        
        # Confident unsafe
        action = policy.decide({"unsafe"}, "benign", 0.0)
        assert action == "block"
    
    def test_hard_override(self):
        """Test Phase T block override."""
        policy = EscalationPolicy(conservative=True)
        
        # Even if UQ says safe, Phase T block wins
        action = policy.decide({"safe"}, "block", 0.9)
        assert action == "block"
    
    def test_conservative_fallback(self):
        """Test conservative fallback for uncertainty."""
        policy = EscalationPolicy(conservative=True, enable_human_review=False)
        
        # Uncertain with no strong signal → steer (not allow)
        action = policy.decide({"safe", "unsafe"}, "benign", 0.0)
        assert action == "steer"
    
    def test_tiebreaker(self):
        """Test Phase T tiebreaker."""
        policy = EscalationPolicy(conservative=True)
        
        # Uncertain + Phase T says mitigate → steer
        action = policy.decide({"safe", "unsafe"}, "mitigate", 0.8)
        assert action == "steer"


class TestUncertaintyFilter:
    """Test uncertainty filter integration."""
    
    def test_passthrough_mode(self):
        """Test disabled UQ (passthrough)."""
        predictor = SplitConformalPredictor(alpha=0.05)
        predictor.calibrate(np.array([0.1, 0.2, 0.3]))
        
        uq_filter = UncertaintyFilter(predictor, enabled=False)
        
        decision = uq_filter.filter(
            risk_score=0.3,
            policy_state="benign",
            escalation_score=0.0
        )
        
        assert decision.action in ["allow", "block"]
        assert decision.set_certainty == 1.0
    
    def test_hard_override(self):
        """Test Phase T block cannot be overridden."""
        predictor = SplitConformalPredictor(alpha=0.05)
        predictor.calibrate(np.array([0.1, 0.2, 0.3]))
        
        uq_filter = UncertaintyFilter(predictor, enabled=True)
        
        decision = uq_filter.filter(
            risk_score=0.1,  # Low risk
            policy_state="block",  # But Phase T blocks
            escalation_score=0.9
        )
        
        assert decision.action == "block"
        assert decision.uncertainty_reason == "Phase T hard block"


class TestMonitoring:
    """Test monitoring and logging."""
    
    def test_logger(self):
        """Test UQ logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "uq.jsonl"
            logger = UQLogger(str(log_path))
            
            # Mock decision
            from src.safety.alrc.phase_uq.integration import UQDecision
            decision = UQDecision(
                action="allow",
                prediction_set={"safe"},
                set_certainty=1.0,
                tau_used=0.3,
                model_version="test",
                groups_active=None,
                timestamp=0.0
            )
            
            context = {
                'session_id': 'test123',
                'risk_score': 0.2,
                'policy_state': 'benign',
                'escalation_score': 0.0,
                'prompt_length': 50
            }
            
            logger.log(decision, context)
            
            # Verify log file created
            assert log_path.exists()
            
            # Verify content
            with open(log_path) as f:
                line = f.readline()
                data = json.loads(line)
                assert data['action'] == 'allow'
                assert data['session_id'] == 'test123'
    
    def test_monitor_metrics(self):
        """Test metric computation."""
        monitor = UQMonitor()
        
        # Mock logs
        logs = [
            {'is_uncertain': False, 'action': 'allow', 'groups_active': [], 'risk_bucket': 'low'},
            {'is_uncertain': True, 'action': 'steer', 'groups_active': ['high_risk'], 'risk_bucket': 'high'},
            {'is_uncertain': False, 'action': 'block', 'groups_active': [], 'risk_bucket': 'high'},
            {'is_uncertain': True, 'action': 'escalate_human', 'groups_active': ['ambiguous'], 'risk_bucket': 'medium'},
        ]
        
        metrics = monitor.compute_metrics(logs)
        
        assert metrics['total_samples'] == 4
        assert metrics['uncertainty_rate'] == 0.5  # 2/4
        assert metrics['escalation_rate'] == 0.25  # 1/4
    
    def test_recalibration_trigger(self):
        """Test recalibration triggers."""
        monitor = UQMonitor()
        
        # High uncertainty rate
        metrics = {'uncertainty_rate': 0.25, 'per_group_uncertainty': {}}
        assert monitor.should_recalibrate(metrics)
        
        # High group uncertainty
        metrics = {'uncertainty_rate': 0.10, 'per_group_uncertainty': {'high_risk': 0.35}}
        assert monitor.should_recalibrate(metrics)
        
        # OK
        metrics = {'uncertainty_rate': 0.10, 'per_group_uncertainty': {'high_risk': 0.15}}
        assert not monitor.should_recalibrate(metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
