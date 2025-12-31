"""Tests for Phase 3: Hardening & Observability.

Run:
    pytest phase_3_tests.py -v
"""

import pytest
import numpy as np
import time

from src.safety.alrc.phase_3 import (
    SafetyController, SafetyState,
    MMDDriftDetector, DriftAttributor,
    OscillationDetector, EconomicRateLimiter,
    GovernanceApproval, SafetyInvariants,
    SafetyWatchdog
)


class TestSafetyController:
    """Test safety controller."""
    
    def test_state_inference(self):
        """Test state inference from signals."""
        controller = SafetyController()
        
        # Normal state
        signals = {'mmd_drift': 0.05, 'oscillation_count': 0, 'uncertainty_rate': 0.1, 'attack_entropy': 0.2}
        state = controller.observe(signals)
        assert state == SafetyState.NORMAL
        
        # Attack state (high drift + directedness)
        signals = {'mmd_drift': 0.25, 'oscillation_count': 0, 'uncertainty_rate': 0.1, 'attack_entropy': 0.8}
        state = controller.observe(signals)
        assert state == SafetyState.ACTIVE_ATTACK
        
        # Benign drift (high drift, low directedness)
        signals = {'mmd_drift': 0.15, 'oscillation_count': 0, 'uncertainty_rate': 0.1, 'attack_entropy': 0.1}
        state = controller.observe(signals)
        assert state == SafetyState.BENIGN_DRIFT
    
    def test_action_selection(self):
        """Test action mapping."""
        controller = SafetyController()
        
        # Normal → monitor
        controller.state = SafetyState.NORMAL
        actions = controller.choose_action(SafetyState.NORMAL)
        assert "monitor" in actions
        
        # Attack → freeze learning
        actions = controller.choose_action(SafetyState.ACTIVE_ATTACK)
        assert "freeze_learning" in actions
        assert "harden_thresholds" in actions
    
    def test_state_transitions(self):
        """Test state transition recording."""
        controller = SafetyController()
        
        signals = {'mmd_drift': 0.25, 'attack_entropy': 0.8}
        controller.transition_state(SafetyState.ACTIVE_ATTACK, "drift_spike", signals)
        
        assert controller.state == SafetyState.ACTIVE_ATTACK
        assert len(controller.state_history) == 1
        assert controller.state_history[0].to_state == SafetyState.ACTIVE_ATTACK


class TestMMDDriftDetector:
    """Test MMD drift detector."""
    
    def test_drift_detection(self):
        """Test drift is detected."""
        # Baseline
        baseline = np.random.randn(1000, 384)
        detector = MMDDriftDetector(baseline, threshold=0.15)
        
        # Similar distribution → no drift
        for _ in range(200):
            embedding = np.random.randn(384)
            mmd = detector.update(embedding)
        
        assert mmd < 0.15, f"Should not drift on similar data: {mmd}"
        
        # Shifted distribution → drift
        detector.reset_window()
        
        for _ in range(200):
            embedding = np.random.randn(384) + 2.0  # Shift
            mmd = detector.update(embedding)
        
        assert mmd > 0.15, f"Should detect drift: {mmd}"
        assert detector.alert_count > 0
    
    def test_velocity(self):
        """Test drift velocity calculation."""
        baseline = np.random.randn(1000, 384)
        detector = MMDDriftDetector(baseline)
        
        # Gradual drift
        for i in range(50):
            shift = i * 0.01
            embedding = np.random.randn(384) + shift
            detector.update(embedding)
        
        velocity = detector.get_velocity()
        assert velocity > 0, "Velocity should be positive for increasing drift"


class TestDriftAttributor:
    """Test drift attribution."""
    
    def test_directedness(self):
        """Test directedness calculation."""
        baseline = np.zeros(10)
        unsafe = np.ones(10) * 5.0  # Clear direction
        
        attributor = DriftAttributor(baseline, unsafe)
        
        # Drift toward unsafe
        window = np.array([np.ones(10) * 3.0 for _ in range(100)])
        directedness = attributor.compute_directedness(window)
        
        assert directedness > 0.7, f"Should detect directedness: {directedness}"
        
        # Random drift
        window = np.random.randn(100, 10)
        directedness = attributor.compute_directedness(window)
        
        assert directedness < 0.6, f"Should not detect directedness in random: {directedness}"
    
    def test_classification(self):
        """Test drift classification."""
        attributor = DriftAttributor()
        
        # Hostile: high velocity, high directedness
        cls = attributor.classify_drift(mmd=0.20, directedness=0.8, velocity=0.1, window_size=1000)
        assert cls == "hostile"
        
        # Benign: slow, low directedness
        cls = attributor.classify_drift(mmd=0.10, directedness=0.2, velocity=0.005, window_size=1000)
        assert cls == "benign"


class TestOscillationDetector:
    """Test oscillation detection."""
    
    def test_lockout(self):
        """Test lockout after threshold."""
        detector = OscillationDetector(enabled=False)  # In-memory fallback
        session = "test_session"
        
        # Simulate rapid refusals (if Redis available)
        if detector.enabled:
            for i in range(6):
                locked = detector.track_refusal(session)
                
                if i < 5:
                    assert not locked
                else:
                    assert locked, "Should lock after 5 refusals"
            
            assert detector.is_locked_out(session)


class TestEconomicRateLimiter:
    """Test economic rate limiter."""
    
    def test_cost_tracking(self):
        """Test cost accumulation."""
        limiter = EconomicRateLimiter(enabled=False)
        
        if limiter.enabled:
            session = "test_session"
            
            # Track costs
            limiter.track_cost(session, "query")
            limiter.track_cost(session, "refusal")
            
            stats = limiter.get_stats(session)
            assert stats['total_cost'] == 6  # 1 + 5
    
    def test_rate_limiting(self):
        """Test rate limit threshold."""
        limiter = EconomicRateLimiter(enabled=False)
        
        if limiter.enabled:
            session = "test_session"
            
            # Exceed threshold
            for _ in range(15):  # 15 refusals = 75 cost
                limiter.track_cost(session, "refusal")
            
            assert limiter.should_rate_limit(session)
            assert limiter.get_wait_time(session) > 0


class TestGovernanceApproval:
    """Test governance approval."""
    
    def test_approval_workflow(self):
        """Test approval request and grant."""
        governance = GovernanceApproval(enabled=True)
        
        # Request
        approval_id = governance.request_approval("adapt_baseline", {"reason": "drift detected"})
        assert not governance.check_approval(approval_id)
        
        # Approve
        governance.approve(approval_id, "admin")
        assert governance.check_approval(approval_id)
    
    def test_auto_approve_disabled(self):
        """Test auto-approve when disabled."""
        governance = GovernanceApproval(enabled=False)
        
        approval_id = governance.request_approval("adapt_baseline", {})
        assert governance.check_approval(approval_id)  # Auto-approved


class TestSafetyInvariants:
    """Test safety invariants."""
    
    def test_invariant_check(self):
        """Test invariant checking."""
        controller = SafetyController()
        
        # All OK
        violations = SafetyInvariants.check_all(controller)
        assert len(violations) == 0
        
        # Violate: learning during attack
        controller.state = SafetyState.ACTIVE_ATTACK
        controller.learning_enabled = True
        
        violations = SafetyInvariants.check_all(controller)
        assert len(violations) > 0
        assert any("Learning enabled" in v for v in violations)
    
    def test_action_verification(self):
        """Test action verification."""
        controller = SafetyController()
        
        # Safe action
        is_safe = SafetyInvariants.verify_action(controller, "monitor")
        assert is_safe
        
        # Unsafe action
        controller.state = SafetyState.ACTIVE_ATTACK
        is_safe = SafetyInvariants.verify_action(controller, "adapt_baseline")
        assert not is_safe


class TestSafetyWatchdog:
    """Test safety watchdog."""
    
    def test_drift_detector_health(self):
        """Test drift detector health check."""
        watchdog = SafetyWatchdog()
        
        baseline = np.random.randn(100, 10)
        detector = MMDDriftDetector(baseline)
        
        # Update detector
        detector.update(np.random.randn(10))
        
        # Check
        healthy = watchdog.check_drift_detector(detector)
        assert healthy
    
    def test_controller_health(self):
        """Test controller health check."""
        watchdog = SafetyWatchdog()
        controller = SafetyController()
        
        healthy = watchdog.check_controller(controller)
        assert healthy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
