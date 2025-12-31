"""Phase UQ integration with ALRC pipeline."""

from dataclasses import dataclass, asdict
from typing import Set, Dict, Optional
import time
import logging

from .conformal import SplitConformalPredictor, ConformalResult
from .mvp_predictor import MVPPredictor
from .mvp_groups import get_groups
from .escalation import EscalationPolicy

logger = logging.getLogger(__name__)


@dataclass
class UQDecision:
    """Final decision from Phase UQ."""
    action: str  # "allow", "block", "steer", "escalate_human"
    prediction_set: Set[str]
    set_certainty: float
    uncertainty_reason: Optional[str] = None
    
    # For logging/monitoring
    tau_used: float = 0.0
    model_version: str = "unknown"
    groups_active: Optional[Set[str]] = None
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dict for logging."""
        d = asdict(self)
        # Convert sets to lists for JSON serialization
        d['prediction_set'] = list(self.prediction_set) if self.prediction_set else []
        d['groups_active'] = list(self.groups_active) if self.groups_active else []
        return d


class UncertaintyFilter:
    """
    Phase UQ: Conformal uncertainty quantification.
    
    Converts risk scores to prediction sets with coverage guarantees.
    Supports both SCP (marginal coverage) and MVP (group-conditional).
    """
    
    def __init__(
        self,
        predictor,  # SplitConformalPredictor or MVPPredictor
        escalation_policy: Optional[EscalationPolicy] = None,
        use_mvp: bool = False,
        enabled: bool = True
    ):
        """
        Args:
            predictor: Calibrated conformal predictor
            escalation_policy: How to handle uncertain predictions
            use_mvp: Use MVP with groups (vs SCP)
            enabled: Enable UQ filtering
        """
        self.predictor = predictor
        self.policy = escalation_policy or EscalationPolicy()
        self.use_mvp = use_mvp
        self.enabled = enabled
        
        logger.info(
            f"UncertaintyFilter initialized: "
            f"enabled={enabled}, use_mvp={use_mvp}"
        )
    
    def filter(
        self,
        risk_score: float,
        policy_state: str,
        escalation_score: float = 0.0,
        query: str = "",
        long_term_drift: float = 0.0,
        session_refusals: int = 0
    ) -> UQDecision:
        """
        Apply UQ filter to pipeline result.
        
        CRITICAL: Phase T block always wins (hard override).
        
        Args:
            risk_score: From Phase A
            policy_state: From Phase T
            escalation_score: From Phase T
            query: User input (for MVP groups)
            long_term_drift: From Phase L
            session_refusals: Recent refusal count
        
        Returns:
            UQDecision with action
        """
        timestamp = time.time()
        
        # HARD SAFETY OVERRIDE
        # Phase T block cannot be overridden
        if policy_state == "block":
            return UQDecision(
                action="block",
                prediction_set={"unsafe"},
                set_certainty=1.0,
                uncertainty_reason="Phase T hard block",
                timestamp=timestamp
            )
        
        if not self.enabled:
            # Passthrough mode
            action = "block" if risk_score > 0.7 else "allow"
            return UQDecision(
                action=action,
                prediction_set={"safe"} if action == "allow" else {"unsafe"},
                set_certainty=1.0,
                timestamp=timestamp
            )
        
        # Convert risk to probs
        probs = {
            "safe": 1.0 - risk_score,
            "unsafe": risk_score
        }
        
        # Get conformal prediction
        if self.use_mvp and isinstance(self.predictor, MVPPredictor):
            # MVP with groups
            groups = get_groups(
                query=query,
                risk_score=risk_score,
                escalation_score=escalation_score,
                long_term_drift=long_term_drift,
                session_refusals=session_refusals
            )
            result = self.predictor.predict_set(probs, groups)
        else:
            # Standard SCP
            groups = None
            result = self.predictor.predict_set(probs)
        
        # Apply escalation policy
        action = self.policy.decide(
            prediction_set=result.prediction_set,
            policy_state=policy_state,
            escalation_score=escalation_score
        )
        
        # Build uncertainty reason
        reason = None
        if result.is_uncertain:
            reason = f"Conformal set={result.prediction_set}, τ={result.tau_used:.3f}"
            if groups:
                reason += f", groups={groups}"
        
        return UQDecision(
            action=action,
            prediction_set=result.prediction_set,
            set_certainty=result.set_certainty,
            uncertainty_reason=reason,
            tau_used=result.tau_used,
            model_version=result.model_version,
            groups_active=groups,
            timestamp=timestamp
        )
    
    def get_stats(self) -> Dict:
        """Get UQ filter statistics."""
        return {
            'enabled': self.enabled,
            'use_mvp': self.use_mvp,
            'predictor_stats': self.predictor.get_stats() if hasattr(self.predictor, 'get_stats') else {},
        }
