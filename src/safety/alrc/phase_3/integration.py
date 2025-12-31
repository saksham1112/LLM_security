"""Phase 3 Integration: Example integration with ALRC pipeline.

This demonstrates how Phase 3 components work together.
"""

import numpy as np
from src.safety.alrc.phase_3 import (
    SafetyController,
    MMDDriftDetector,
    DriftAttributor,
    OscillationDetector,
    EconomicRateLimiter,
    GovernanceApproval,
    SafetyInvariants,
    SafetyWatchdog,
)


class Phase3Integration:
    """
    Phase 3 integration wrapper.
    
    Coordinates all Phase 3 components.
    """
    
    def __init__(
        self,
        baseline_embeddings: np.ndarray,
        unsafe_embeddings: np.ndarray = None,
        redis_url: str = "redis://localhost:6379",
        enable_governance: bool = True
    ):
        # Control
        self.controller = SafetyController()
        
        # Drift
        self.drift_detector = MMDDriftDetector(baseline_embeddings)
        self.drift_attributor = DriftAttributor(
            baseline_centroid=np.mean(baseline_embeddings, axis=0),
            unsafe_centroid=np.mean(unsafe_embeddings, axis=0) if unsafe_embeddings is not None else None
        )
        
        # Attack defense
        self.oscillation_detector = OscillationDetector(redis_url)
        self.economic_limiter = EconomicRateLimiter(redis_url)
        
        # Governance
        self.governance = GovernanceApproval(enabled=enable_governance)
        
        # Watchdog
        self.watchdog = SafetyWatchdog()
        
    def pre_flight_check(self, session_id: str) -> dict:
        """
        Pre-flight checks before processing request.
        
        Returns:
            {"allow": bool, "reason": str}
        """
        # Check lockout
        if self.oscillation_detector.is_locked_out(session_id):
            return {"allow": False, "reason": "Session locked out (oscillation detected)"}
        
        # Check economic limit
        if self.economic_limiter.should_rate_limit(session_id):
            wait_time = self.economic_limiter.get_wait_time(session_id)
            return {"allow": False, "reason": f"Rate limit exceeded. Wait {wait_time}s"}
        
        return {"allow": True, "reason": "OK"}
    
    def update(self, embedding: np.ndarray, session_id: str, action: str, is_uncertain: bool):
        """
        Update Phase 3 components with new request.
        
        Args:
            embedding: Query embedding
            session_id: Session ID
            action: Pipeline action ("allow", "block", "steer")
            is_uncertain: Whether UQ was uncertain
        """
        # Track costs
        self.economic_limiter.track_cost(session_id, "query")
        
        if action in ["block", "steer"]:
            self.economic_limiter.track_cost(session_id, "refusal")
            
            # Track oscillation
            locked = self.oscillation_detector.track_refusal(session_id)
            if locked:
                self.economic_limiter.track_cost(session_id, "lockout")
        
        if is_uncertain:
            self.economic_limiter.track_cost(session_id, "uncertain")
        
        # Update drift
        mmd = self.drift_detector.update(embedding)
        
        # Compute signals
        signals = self.compute_signals(session_id)
        
        # Controller step
        decision = self.controller.step(signals)
        
        # Check invariants
        SafetyInvariants.enforce(self.controller)
        
        return decision
    
    def compute_signals(self, session_id: str) -> dict:
        """Compute signals for controller."""
        drift_stats = self.drift_detector.get_stats()
        mmd = drift_stats['mmd_current']
        velocity = drift_stats['velocity']
        
        # Attribution
        if len(self.drift_detector.live_window) > 0:
            window_array = np.array(list(self.drift_detector.live_window))
            directedness = self.drift_attributor.compute_directedness(window_array)
            drift_class = self.drift_attributor.classify_drift(
                mmd, directedness, velocity, len(self.drift_detector.live_window)
            )
        else:
            directedness = 0.0
            drift_class = "benign"
        
        # Oscillation
        osc_stats = self.oscillation_detector.get_stats(session_id)
        oscillation_count = osc_stats.get('refusal_count', 0)
        
        return {
            'mmd_drift': mmd,
            'velocity': velocity,
            'attack_entropy': directedness,
            'drift_classification': drift_class,
            'oscillation_count': oscillation_count,
            'uncertainty_rate': 0.0,  # Would come from UQ logger
        }
    
    def health_check(self, pipeline) -> dict:
        """Run watchdog health check."""
        return self.watchdog.watchdog_loop(pipeline)


# Example usage
if __name__ == "__main__":
    # Initialize with dummy data
    baseline = np.random.randn(1000, 384)
    unsafe = np.random.randn(100, 384) + 2.0  # Shifted distribution
    
    phase3 = Phase3Integration(baseline, unsafe, enable_governance=False)
    
    # Simulate request
    session = "demo_session"
    
    # Pre-flight
    check = phase3.pre_flight_check(session)
    print(f"Pre-flight: {check}")
    
    # Process
    embedding = np.random.randn(384)
    decision = phase3.update(embedding, session, action="allow", is_uncertain=False)
    
    print(f"Controller state: {decision['state']}")
    print(f"Proposed actions: {decision['proposed_actions']}")
