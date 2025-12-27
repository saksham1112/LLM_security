"""
Session Risk Floor - Stateful Suspicion Mechanism
Prevents "Topic Reset" jailbreaks by maintaining a risk floor after high-risk turns.

Attack Pattern Defended:
1. User escalates to high risk (Risk=1.0)
2. User inserts "semantic spacer" ("let's leave this")
3. User makes harmful request with softened framing
4. Vector search only sees benign recent context → BYPASS

Defense:
- If any turn exceeds threshold (e.g., 0.8), set persistent floor (e.g., 0.4)
- Floor decays slowly (30-min half-life)
- Final_Risk = Max(Current_Risk, Session_Risk_Floor)
"""

import time
import math
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionRiskFloor:
    """Stateful risk floor for a session."""
    max_risk: float  # Highest risk ever seen
    floor_risk: float  # Current active floor (with decay)
    last_update: float  # Unix timestamp of last update
    trigger_count: int  # How many times high-risk threshold was hit


class RiskFloorManager:
    """
    Manages persistent session risk floors to prevent topic reset attacks.
    
    Mechanism:
    - When Risk > TRIGGER_THRESHOLD (0.8), latch a floor at FLOOR_MULTIPLIER * max_risk
    - Floor decays with HALF_LIFE (30 minutes by default)
    - Prevents users from "resetting" context with benign statements
    """
    
    # Configuration
    TRIGGER_THRESHOLD = 0.8  # Risk must exceed this to latch floor
    FLOOR_MULTIPLIER = 0.6  # Floor set to 60% of max risk (was 0.5)
    HALF_LIFE_SECONDS = 1800  # 30 minutes decay half-life
    MIN_FLOOR = 0.15  # Floor can't decay below this if ever triggered
    
    def __init__(self):
        """Initialize risk floor manager."""
        # In-memory storage (should be Redis in production)
        self._floors: Dict[str, SessionRiskFloor] = {}
        logger.info("RiskFloorManager initialized")
    
    def update_floor(
        self,
        session_id: str,
        current_risk: float
    ) -> float:
        """
        Update session risk floor based on current risk.
        
        Args:
            session_id: Session identifier
            current_risk: Current turn's risk score (before floor application)
            
        Returns:
            Active floor value (after decay)
        """
        now = time.time()
        
        # Get or create floor state
        if session_id not in self._floors:
            self._floors[session_id] = SessionRiskFloor(
                max_risk=0.0,
                floor_risk=0.0,
                last_update=now,
                trigger_count=0
            )
        
        floor = self._floors[session_id]
        
        # Apply decay to existing floor
        time_delta = now - floor.last_update
        if time_delta > 0 and floor.floor_risk > 0:
            # Exponential decay: floor(t) = floor(0) * (0.5)^(t/half_life)
            decay_factor = math.pow(0.5, time_delta / self.HALF_LIFE_SECONDS)
            floor.floor_risk = max(
                self.MIN_FLOOR if floor.trigger_count > 0 else 0.0,
                floor.floor_risk * decay_factor
            )
        
        # Check if current risk triggers new floor
        if current_risk > self.TRIGGER_THRESHOLD:
            # Update max risk
            if current_risk > floor.max_risk:
                floor.max_risk = current_risk
            
            # Set new floor at FLOOR_MULTIPLIER * max_risk
            new_floor = self.FLOOR_MULTIPLIER * floor.max_risk
            
            # Take max of current floor and new floor (ratchet up)
            if new_floor > floor.floor_risk:
                floor.floor_risk = new_floor
                floor.trigger_count += 1
                
                logger.warning(
                    f"Session {session_id[:8]}... triggered risk floor: "
                    f"current_risk={current_risk:.2f}, "
                    f"new_floor={new_floor:.2f}, "
                    f"trigger_count={floor.trigger_count}"
                )
        
        # Update timestamp
        floor.last_update = now
        
        return floor.floor_risk
    
    def apply_floor(
        self,
        session_id: str,
        current_risk: float
    ) -> tuple[float, float, bool]:
        """
        Apply session risk floor to current risk.
        
        Args:
            session_id: Session identifier
            current_risk: Calculated risk before floor
            
        Returns:
            (final_risk, floor_value, floor_applied)
        """
        # Update floor based on current risk
        floor_value = self.update_floor(session_id, current_risk)
        
        # Apply floor (take maximum)
        final_risk = max(current_risk, floor_value)
        floor_applied = final_risk > current_risk
        
        if floor_applied:
            logger.info(
                f"Session {session_id[:8]}... risk floor applied: "
                f"{current_risk:.2f} → {final_risk:.2f} "
                f"(floor={floor_value:.2f})"
            )
        
        return final_risk, floor_value, floor_applied
    
    def get_floor_info(self, session_id: str) -> Optional[SessionRiskFloor]:
        """Get current floor information for a session."""
        return self._floors.get(session_id)
    
    def reset_floor(self, session_id: str):
        """Reset floor for a session (admin override)."""
        if session_id in self._floors:
            del self._floors[session_id]
            logger.info(f"Reset risk floor for session {session_id[:8]}...")


# Quick test
if __name__ == "__main__":
    import time
    
    manager = RiskFloorManager()
    session = "test_topic_reset"
    
    print("\n" + "="*70)
    print("SESSION RISK FLOOR TEST - Topic Reset Defense")
    print("="*70)
    
    # Simulate the exact attack sequence
    turns = [
        (0.13, "Turn 1: Benign start"),
        (0.09, "Turn 2: Still benign"),
        (0.18, "Turn 3: Slight escalation"),
        (0.33, "Turn 4: Yellow zone"),
        (1.0, "Turn 5: HIGH RISK - triggers floor"),
        (0.71, "Turn 6: Still risky"),
        (0.53, "Turn 7: Moderate"),
        (0.51, "Turn 8: Moderate"),
        (0.21, "Turn 9: TOPIC RESET attempt"),
        (0.18, "Turn 10: ATTACK - 'toxic gas to scare'"),
    ]
    
    for i, (risk, desc) in enumerate(turns):
        final_risk, floor_val, applied = manager.apply_floor(session, risk)
        
        status = "🔴 FLOOR BLOCKED" if applied and final_risk > 0.5 else \
                 "🟡 FLOOR RAISED" if applied else \
                 "🟢 NORMAL"
        
        print(f"\n{desc}")
        print(f"  {status}")
        print(f"  Calculated Risk: {risk:.2f}")
        print(f"  Floor Value: {floor_val:.2f}")
        print(f"  Final Risk: {final_risk:.2f}")
        
        # Simulate time passing
        time.sleep(0.1)
    
    # Test decay
    print("\n" + "="*70)
    print("Testing Decay (simulating 15 minutes)")
    print("="*70)
    
    # Simulate 15 minutes passing
    floor_info = manager._floors[session]
    floor_info.last_update -= 900  # 15 minutes ago
    
    final_risk, floor_val, applied = manager.apply_floor(session, 0.15)
    print(f"\nAfter 15 min decay:")
    print(f"  Floor: {floor_val:.2f} (should be ~0.35, half of 0.50)")
    print(f"  Input Risk: 0.15")
    print(f"  Final Risk: {final_risk:.2f}")
    
    # Verify Turn 10 would be blocked
    print("\n" + "="*70)
    print("VERIFICATION: Turn 10 (Topic Reset Attack)")
    print("="*70)
    turn_10_risk = 0.18
    final, floor, blocked = manager.apply_floor(session, turn_10_risk)
    
    if blocked and final > 0.5:
        print(f"✅ ATTACK BLOCKED!")
        print(f"   Input Risk: {turn_10_risk:.2f} (would be GREEN)")
        print(f"   Floor Applied: {floor:.2f}")
        print(f"   Final Risk: {final:.2f} (now YELLOW/RED)")
    else:
        print(f"❌ ATTACK SUCCEEDED")
        print(f"   Final Risk: {final:.2f}")
