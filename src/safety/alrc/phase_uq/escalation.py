"""Escalation policy for uncertain predictions."""

from typing import Set
import logging

logger = logging.getLogger(__name__)


class EscalationPolicy:
    """
    Decision policy when prediction set is uncertain.
    
    Conservative by default: uncertain → steer/block (not allow).
    """
    
    def __init__(
        self, 
        enable_human_review: bool = False,
        conservative: bool = True
    ):
        """
        Args:
            enable_human_review: Enable escalation to human moderation
            conservative: Use conservative fallback (True in production)
        """
        self.enable_human_review = enable_human_review
        self.conservative = conservative
        
        logger.info(
            f"EscalationPolicy: human_review={enable_human_review}, "
            f"conservative={conservative}"
        )
    
    def decide(
        self,
        prediction_set: Set[str],
        policy_state: str,
        escalation_score: float
    ) -> str:
        """
        Decide action based on prediction set and Phase T signals.
        
        CRITICAL: Phase T block = hard override (always block).
        
        Args:
            prediction_set: From conformal predictor
            policy_state: From Phase T
            escalation_score: From Phase T
        
        Returns:
            action: "allow", "block", "steer", "escalate_human"
        """
        
        # HARD SAFETY OVERRIDE
        # Phase T block cannot be overridden by UQ
        if policy_state == "block":
            logger.debug("Hard override: Phase T blocked")
            return "block"
        
        # Confident prediction
        if len(prediction_set) == 1:
            if "unsafe" in prediction_set:
                return "block"
            else:
                return "allow"
        
        # UNCERTAIN: {safe, unsafe}
        # Use Phase T as tiebreaker
        if policy_state in ["mitigate", "steer"]:
            logger.debug(f"Uncertain → using Phase T: {policy_state}")
            return "steer"
        
        if escalation_score > 0.5:
            logger.debug(f"Uncertain → high escalation ({escalation_score:.2f}) → steer")
            return "steer"
        
        # Still uncertain with no strong signal
        if self.enable_human_review:
            logger.debug("Uncertain → escalating to human")
            return "escalate_human"
        else:
            # CONSERVATIVE FALLBACK
            if self.conservative:
                logger.debug("Uncertain → conservative fallback → steer")
                return "steer"
            else:
                # Permissive (dev/testing only)
                logger.warning("Uncertain → permissive fallback → allow")
                return "allow"
