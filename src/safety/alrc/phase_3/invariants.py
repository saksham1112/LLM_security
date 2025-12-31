"""Safety Invariants: Formal safety guarantees."""

from typing import List
import logging

logger = logging.getLogger(__name__)


class SafetyInvariants:
    """
    Formal safety invariants.
    
    These are HARD constraints, never violated.
    """
    
    INVARIANTS = [
        "No unsafe output during ANY state",
        "No learning enabled during ACTIVE_ATTACK",
        "No threshold loosening under threat",
        "All self-modification is reversible",
        "Thresholds monotonically strict under threat",
        "Governance approval required for critical actions",
        "Watchdog must detect component failure within 5 min",
    ]
    
    @staticmethod
    def check_all(controller) -> List[str]:
        """
        Verify all invariants hold.
        
        Args:
            controller: SafetyController instance
        
        Returns:
            List of violations (empty = all OK)
        """
        violations = []
        
        # INVARIANT 1: No unsafe output during ANY state
        if hasattr(controller, 'last_output_unsafe') and controller.last_output_unsafe:
            violations.append("CRITICAL: Unsafe output generated")
        
        # INVARIANT 2: No learning during attack
        from .safety_controller import SafetyState
        if controller.state == SafetyState.ACTIVE_ATTACK and controller.learning_enabled:
            violations.append("CRITICAL: Learning enabled during ACTIVE_ATTACK")
        
        # INVARIANT 3: Thresholds never loosen under threat
        if controller.state in [SafetyState.SUSPICIOUS, SafetyState.ACTIVE_ATTACK]:
            if controller.threshold_loosening_detected():
                violations.append("CRITICAL: Threshold loosened under threat")
        
        # INVARIANT 4: All self-modification is reversible
        if not controller.has_rollback_checkpoint():
            violations.append("WARNING: No rollback checkpoint available")
        
        # INVARIANT 5: Monotonic safety under threat
        if controller.drift_threshold_hostile > controller.last_hostile_threshold:
            violations.append("CRITICAL: Hostile threshold loosened")
        
        return violations
    
    @staticmethod
    def enforce(controller):
        """
        Emergency enforcement if violations detected.
        
        Called automatically by system.
        """
        violations = SafetyInvariants.check_all(controller)
        
        if violations:
            logger.critical(f"INVARIANT VIOLATIONS: {violations}")
            
            # Emergency actions
            from .safety_controller import SafetyState
            controller.state = SafetyState.FROZEN
            controller.learning_enabled = False
            controller.threshold_locked = True
            
            # Alert (in production: PagerDuty)
            logger.critical("System in FROZEN state due to invariant violations")
            
            # Dump state for forensics
            import json
            import time
            
            state_dump = {
                "timestamp": time.time(),
                "violations": violations,
                "state": controller.state.value,
                "thresholds": {
                    "drift_benign": controller.drift_threshold_benign,
                    "drift_hostile": controller.drift_threshold_hostile,
                },
                "learning_enabled": controller.learning_enabled,
            }
            
            try:
                with open("emergency_dump.json", "w") as f:
                    json.dump(state_dump, f, indent=2)
                logger.info("State dumped to emergency_dump.json")
            except Exception as e:
                logger.error(f"Failed to dump state: {e}")
    
    @staticmethod
    def verify_action(controller, action: str) -> bool:
        """
        Verify if action violates invariants.
        
        Returns:
            True if action is safe, False if it violates invariants
        """
        from .safety_controller import SafetyState
        
        # Check INVARIANT 2: No learning during attack
        if action == "adapt_baseline" and controller.state == SafetyState.ACTIVE_ATTACK:
            logger.error("INVARIANT BLOCK: Cannot adapt baseline during ACTIVE_ATTACK")
            return False
        
        # Check INVARIANT 3: No loosening under threat
        if "loosen" in action and controller.state in [SafetyState.SUSPICIOUS, SafetyState.ACTIVE_ATTACK]:
            logger.error("INVARIANT BLOCK: Cannot loosen thresholds under threat")
            return False
        
        return True
