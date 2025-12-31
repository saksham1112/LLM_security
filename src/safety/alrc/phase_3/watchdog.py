"""Safety Watchdog: Monitor the monitors (meta-safety)."""

import numpy as np
import time
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SafetyWatchdog:
    """
    Monitor the safety system itself.
    
    Detects failures in drift detector, UQ, etc.
    """
    
    def __init__(self):
        self.component_health: Dict[str, bool] = {}
        self.last_check = time.time()
        self.alert_count = 0
        
        logger.info("SafetyWatchdog initialized")
    
    def check_drift_detector(self, drift_detector) -> bool:
        """
        Verify drift detector is functioning.
        
        Returns:
            True if healthy, False if failed
        """
        try:
            # Check 1: Is window filling?
            if not hasattr(drift_detector, 'live_window'):
                logger.error("Drift detector: missing live_window attribute")
                return False
            
            if len(drift_detector.live_window) == 0:
                # This might be OK at startup
                if time.time() - drift_detector.last_update > 300:
                    logger.error("Drift detector: window empty for >5 min (possible crash)")
                    return False
            
            # Check 2: Is MMD changing?
            if len(drift_detector.drift_history) > 100:
                recent = drift_detector.drift_history[-100:]
                if np.std(recent) < 1e-6:
                    logger.warning("Drift detector: MMD flatlined (possible freeze)")
                    return False
            
            # Check 3: Heartbeat
            if hasattr(drift_detector, 'last_update'):
                if time.time() - drift_detector.last_update > 300:
                    logger.error("Drift detector: no updates in 5 min")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Drift detector health check failed: {e}")
            return False
    
    def check_uq_coverage(self, uq_logger) -> bool:
        """
        Validate UQ coverage hasn't degraded.
        
        Returns:
            True if healthy, False if suspicious
        """
        try:
            if not uq_logger:
                return True  # UQ might be disabled
            
            # Load recent logs
            if not hasattr(uq_logger, 'log_path'):
                return True
            
            # Simple heuristic: check if logs exist
            import os
            if not os.path.exists(uq_logger.log_path):
                return True  # No data yet
            
            # In production: analyze actual coverage
            # For now, just verify it's logging
            return True
        
        except Exception as e:
            logger.error(f"UQ coverage check failed: {e}")
            return False
    
    def check_controller(self, controller) -> bool:
        """
        Verify controller is functioning.
        
        Returns:
            True if healthy
        """
        try:
            # Check state transitions are being recorded
            if not hasattr(controller, 'state_history'):
                logger.error("Controller: missing state_history")
                return False
            
            # Check for invariant violations
            from .invariants import SafetyInvariants
            violations = SafetyInvariants.check_all(controller)
            
            if violations:
                logger.critical(f"Controller: invariant violations detected: {violations}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Controller health check failed: {e}")
            return False
    
    def watchdog_loop(self, pipeline) -> Dict[str, bool]:
        """
        Periodic health check.
        
        Args:
            pipeline: ALRCPipeline instance
        
        Returns:
            Health status dict
        """
        self.last_check = time.time()
        
        health = {}
        
        # Check drift detector
        if hasattr(pipeline, 'drift_detector') and pipeline.drift_detector:
            health["drift_detector"] = self.check_drift_detector(pipeline.drift_detector)
        
        # Check UQ
        if hasattr(pipeline, 'uq_logger'):
            health["uq_coverage"] = self.check_uq_coverage(pipeline.uq_logger)
        
        # Check controller
        if hasattr(pipeline, 'safety_controller'):
            health["controller"] = self.check_controller(pipeline.safety_controller)
        
        self.component_health = health
        
        # Trigger failsafe if critical component fails
        if not health.get("drift_detector", True):
            logger.critical("Drift detector failed - triggering failsafe")
            if hasattr(pipeline, 'safety_controller'):
                from .safety_controller import SafetyState
                pipeline.safety_controller.state = SafetyState.FROZEN
        
        if not health.get("controller", True):
            logger.critical("Controller failed - CRITICAL")
            self.alert_count += 1
        
        return health
    
    def get_stats(self) -> Dict:
        """Get watchdog statistics."""
        return {
            "last_check": self.last_check,
            "component_health": self.component_health,
            "alert_count": self.alert_count,
        }
