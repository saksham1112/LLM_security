"""Safety Controller: Closed-loop control architecture.

Implements state machine for governed safety decisions.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


class SafetyState(Enum):
    """System safety states."""
    NORMAL = "normal"
    BENIGN_DRIFT = "benign_drift"
    SUSPICIOUS = "suspicious"
    ACTIVE_ATTACK = "active_attack"
    DEGRADED = "degraded"
    FROZEN = "frozen"


@dataclass
class StateTransition:
    """Record of state transition."""
    timestamp: float
    from_state: SafetyState
    to_state: SafetyState
    reason: str
    signals: Dict


class SafetyController:
    """
    Closed-loop safety control system.
    
    Observe → Infer State → Choose Actions → Enforce Invariants → Execute
    """
    
    def __init__(
        self,
        drift_threshold_benign: float = 0.10,
        drift_threshold_hostile: float = 0.20,
        oscillation_threshold: int = 5,
        uncertainty_threshold: float = 0.25
    ):
        self.state = SafetyState.NORMAL
        self.state_history: List[StateTransition] = []
        
        # Thresholds (tunable)
        self.drift_threshold_benign = drift_threshold_benign
        self.drift_threshold_hostile = drift_threshold_hostile
        self.oscillation_threshold = oscillation_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Control flags
        self.learning_enabled = True
        self.threshold_locked = False
        
        # Tracking
        self.last_output_unsafe = False
        self.last_hostile_threshold = drift_threshold_hostile
        
        # Governance
        self.pending_actions: List[str] = []
        self.approved_actions: set = set()
        
        logger.info("SafetyController initialized")
    
    def observe(self, signals: Dict) -> SafetyState:
        """
        Infer system state from signals.
        
        Args:
            signals: {
                'mmd_drift': float,
                'oscillation_count': int,
                'uncertainty_rate': float,
                'attack_entropy': float,  # Directedness metric
                'recent_blocks': int,
            }
        
        Returns:
            Inferred SafetyState
        """
        mmd = signals.get('mmd_drift', 0.0)
        osc = signals.get('oscillation_count', 0)
        unc = signals.get('uncertainty_rate', 0.0)
        entropy = signals.get('attack_entropy', 0.0)
        
        # State inference logic
        if mmd > self.drift_threshold_hostile and entropy > 0.7:
            # High drift + directedness → attack
            return SafetyState.ACTIVE_ATTACK
        
        elif mmd > self.drift_threshold_benign and entropy < 0.3:
            # Moderate drift + low directedness → benign shift
            return SafetyState.BENIGN_DRIFT
        
        elif osc >= self.oscillation_threshold:
            # Rapid probing
            return SafetyState.ACTIVE_ATTACK
        
        elif unc > self.uncertainty_threshold:
            # High uncertainty
            return SafetyState.SUSPICIOUS
        
        elif signals.get('circuit_breaker_open', False):
            return SafetyState.DEGRADED
        
        elif mmd < self.drift_threshold_benign:
            return SafetyState.NORMAL
        
        else:
            return SafetyState.SUSPICIOUS
    
    def choose_action(self, state: SafetyState) -> List[str]:
        """
        Map state to actions.
        
        Returns:
            List of action commands
        """
        actions = []
        
        if state == SafetyState.NORMAL:
            actions.append("monitor")
        
        elif state == SafetyState.BENIGN_DRIFT:
            if self.learning_enabled:
                actions.append("adapt_baseline")
            actions.append("log_drift_event")
        
        elif state == SafetyState.SUSPICIOUS:
            actions.append("increase_logging")
            actions.append("tighten_thresholds_temporary")
        
        elif state == SafetyState.ACTIVE_ATTACK:
            actions.append("freeze_learning")
            actions.append("harden_thresholds")
            actions.append("isolate_sessions")
            actions.append("alert_security_team")
        
        elif state == SafetyState.DEGRADED:
            actions.append("circuit_breaker_active")
            actions.append("reject_new_requests")
        
        elif state == SafetyState.FROZEN:
            actions.append("await_manual_recovery")
        
        return actions
    
    def has_approval(self, action: str) -> bool:
        """Check if action has been approved."""
        return action in self.approved_actions
    
    def approve_action(self, action: str, approver: str = "system"):
        """Grant approval for action."""
        self.approved_actions.add(action)
        logger.info(f"Action '{action}' approved by {approver}")
    
    def threshold_loosening_detected(self) -> bool:
        """Detect if thresholds have been loosened."""
        return self.drift_threshold_hostile > self.last_hostile_threshold
    
    def has_rollback_checkpoint(self) -> bool:
        """Check if we can rollback."""
        # Simplified: assume we always have checkpoint
        return True
    
    def transition_state(self, new_state: SafetyState, reason: str, signals: Dict):
        """Record state transition."""
        if new_state != self.state:
            transition = StateTransition(
                timestamp=time.time(),
                from_state=self.state,
                to_state=new_state,
                reason=reason,
                signals=signals.copy()
            )
            
            self.state_history.append(transition)
            
            logger.info(
                f"State transition: {self.state.value} → {new_state.value} "
                f"(reason: {reason})"
            )
            
            self.state = new_state
    
    def step(self, signals: Dict) -> Dict:
        """
        Main control loop iteration.
        
        Returns:
            Control decision with actions taken
        """
        # 1. Observe
        new_state = self.observe(signals)
        
        # 2. State transition
        if new_state != self.state:
            reason = self._infer_reason(signals)
            self.transition_state(new_state, reason, signals)
        
        # 3. Choose actions
        actions = self.choose_action(self.state)
        
        # 4. Return decision (enforcement happens externally)
        return {
            'state': self.state,
            'proposed_actions': actions,
            'signals': signals,
            'timestamp': time.time()
        }
    
    def _infer_reason(self, signals: Dict) -> str:
        """Infer reason for state change."""
        mmd = signals.get('mmd_drift', 0.0)
        osc = signals.get('oscillation_count', 0)
        unc = signals.get('uncertainty_rate', 0.0)
        
        reasons = []
        
        if mmd > self.drift_threshold_hostile:
            reasons.append(f"high_drift({mmd:.3f})")
        
        if osc > self.oscillation_threshold:
            reasons.append(f"oscillation({osc})")
        
        if unc > self.uncertainty_threshold:
            reasons.append(f"uncertainty({unc:.2%})")
        
        return ", ".join(reasons) if reasons else "threshold_check"
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            'current_state': self.state.value,
            'learning_enabled': self.learning_enabled,
            'threshold_locked': self.threshold_locked,
            'state_transitions': len(self.state_history),
            'pending_actions': len(self.pending_actions),
        }
