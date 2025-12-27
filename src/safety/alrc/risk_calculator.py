"""
Risk Calculator: Final Aggregation Layer
Non-linear risk composition with amplification and momentum.

Purpose: Combine all ALRC layers into single 0-1 risk score.
Latency: <1ms (pure math)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class RiskZone(Enum):
    """Risk zone classification."""
    GREEN = "green"    # 0.0-0.35: Safe, no restrictions
    YELLOW = "yellow"  # 0.35-0.65: Steering, provide safe alternatives
    RED = "red"        # 0.65-1.0: Block or clarify intent


@dataclass
class RiskComponents:
    """Individual risk components from each layer."""
    lexical: float = 0.0       # L1: Keyword-based
    adaptive: float = 0.0      # L2: Statistical anomaly
    short_term: float = 0.0    # L3: Session escalation
    long_term: float = 0.0     # L4: Topic reactivation
    semantic: float = 0.0      # L5: Deep semantic
    
    # Penalties
    escalation_penalty: float = 0.0
    reactivation_penalty: float = 0.0
    
    # Metadata
    matched_keywords: List[str] = None
    matched_topic: Optional[str] = None
    context_type: str = "general"


@dataclass
class RiskResult:
    """Final risk calculation result."""
    final_risk: float
    zone: RiskZone
    components: RiskComponents
    confidence: float  # 0-1, how confident in this assessment
    
    # Recommended action
    action: str  # "allow", "steer", "clarify", "block"
    steering_message: Optional[str] = None


class RiskCalculator:
    """
    Final risk aggregation using non-linear composition.
    
    Features:
    - Weighted component aggregation
    - Non-linear amplification for high-risk scenarios
    - Momentum-adjusted thresholds
    - Zone-based action recommendations
    """
    
    # Component weights (normalized to sum to 1.0)
    WEIGHTS = {
        "lexical": 0.20,     # Fast keyword detection
        "adaptive": 0.10,    # Statistical anomaly
        "short_term": 0.20,  # Session escalation  
        "long_term": 0.15,   # Topic reactivation
        "semantic": 0.35,    # Deep understanding
    }
    
    # Zone thresholds
    GREEN_MAX = 0.35
    YELLOW_MAX = 0.65
    
    # Amplification parameters
    CRITICAL_THRESHOLD = 0.6  # Above this, apply amplification
    AMPLIFICATION_FACTOR = 0.5  # Max amplification
    
    def __init__(self):
        """Initialize the risk calculator."""
        self._risk_history: Dict[str, List[float]] = {}
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth transitions."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _calculate_amplification(self, max_component: float) -> float:
        """
        Calculate critical risk amplification.
        
        If any single component is very high, amplify total risk.
        """
        if max_component <= self.CRITICAL_THRESHOLD:
            return 1.0
        
        # Smooth transition using sigmoid
        x = 10 * (max_component - self.CRITICAL_THRESHOLD)
        amplification = 1.0 + self.AMPLIFICATION_FACTOR * self._sigmoid(x)
        
        return amplification
    
    def _calculate_escalation_multiplier(self, short_term: float) -> float:
        """
        Calculate escalation multiplier based on session pattern.
        
        If user is actively escalating, increase sensitivity.
        """
        if short_term <= 0.15:  # Lowered from 0.2 for earlier trigger
            return 1.0
        
        # More aggressive multiplier for Crescendo attacks
        multiplier = 1.0 + 0.5 * ((short_term - 0.15) / 0.35)
        return min(2.0, multiplier)  # Increased max from 1.5 to 2.0
    
    def _calculate_momentum(self, session_id: str, current_risk: float) -> float:
        """
        Calculate risk momentum (trend).
        
        Rising risk = higher momentum = lower thresholds.
        """
        if session_id not in self._risk_history:
            self._risk_history[session_id] = []
        
        history = self._risk_history[session_id]
        history.append(current_risk)
        
        # Keep last 10 values
        if len(history) > 10:
            history = history[-10:]
            self._risk_history[session_id] = history
        
        if len(history) < 3:
            return 0.0
        
        # Calculate linear regression slope
        x = np.arange(len(history))
        slope = np.polyfit(x, history, 1)[0]
        
        # Normalize to 0-1 range
        momentum = self._sigmoid(5 * slope)
        
        return momentum
    
    def _determine_zone(self, risk: float, momentum: float) -> RiskZone:
        """Determine risk zone with momentum-adjusted thresholds."""
        # Lower thresholds if momentum is positive (rising risk)
        adjustment = 0.05 * momentum
        
        green_threshold = self.GREEN_MAX - adjustment
        yellow_threshold = self.YELLOW_MAX - adjustment
        
        if risk < green_threshold:
            return RiskZone.GREEN
        elif risk < yellow_threshold:
            return RiskZone.YELLOW
        else:
            return RiskZone.RED
    
    def _determine_action(
        self,
        zone: RiskZone,
        components: RiskComponents
    ) -> tuple:
        """Determine action and steering message."""
        if zone == RiskZone.GREEN:
            return "allow", None
        
        elif zone == RiskZone.YELLOW:
            # Provide steering based on detected risk type
            if components.semantic > 0.4:
                message = (
                    "I can help with general information about this topic. "
                    "Could you tell me more about what you're trying to learn?"
                )
            elif components.long_term > 0.3:
                message = (
                    "This seems related to something we discussed earlier. "
                    "Let me provide some general context without implementation details."
                )
            elif components.short_term > 0.3:
                message = (
                    "I notice we're moving toward more sensitive territory. "
                    "Let me explain the concepts without providing specific how-to information."
                )
            else:
                message = (
                    "I can help with this topic at a conceptual level. "
                    "What specifically would you like to understand?"
                )
            return "steer", message
        
        else:  # RED
            if components.lexical > 0.8:
                # Clear block for explicit harmful content
                return "block", "I can't help with that request."
            else:
                # Clarification for borderline cases
                message = (
                    "I want to make sure I understand your question correctly. "
                    "Could you provide more context about what you're trying to achieve? "
                    "This will help me give you the most appropriate response."
                )
                return "clarify", message
    
    def calculate(
        self,
        components: RiskComponents,
        session_id: str = "default"
    ) -> RiskResult:
        """
        Calculate final risk score from components.
        
        Args:
            components: Risk scores from each layer
            session_id: For momentum tracking
            
        Returns:
            RiskResult with final score, zone, and action
        """
        # === Base Weighted Aggregation ===
        base_risk = (
            components.lexical * self.WEIGHTS["lexical"] +
            components.adaptive * self.WEIGHTS["adaptive"] +
            components.short_term * self.WEIGHTS["short_term"] +
            components.long_term * self.WEIGHTS["long_term"] +
            components.semantic * self.WEIGHTS["semantic"]
        )
        
        # Add direct penalties
        base_risk += components.escalation_penalty
        base_risk += components.reactivation_penalty
        
        # === Non-Linear Amplification ===
        max_component = max(
            components.lexical,
            components.semantic,
            components.short_term + components.long_term
        )
        amplification = self._calculate_amplification(max_component)
        
        # === Escalation Multiplier ===
        escalation_mult = self._calculate_escalation_multiplier(components.short_term)
        
        # === Final Risk ===
        final_risk = base_risk * amplification * escalation_mult
        final_risk = min(1.0, max(0.0, final_risk))
        
        # === Momentum ===
        momentum = self._calculate_momentum(session_id, final_risk)
        
        # === Zone Determination ===
        zone = self._determine_zone(final_risk, momentum)
        
        # === Action Determination ===
        action, steering_message = self._determine_action(zone, components)
        
        # === Confidence ===
        # Higher if components agree, lower if they contradict
        component_values = [
            components.lexical, components.semantic,
            components.short_term, components.long_term
        ]
        confidence = 1.0 - np.std(component_values) * 2
        confidence = min(1.0, max(0.3, confidence))
        
        return RiskResult(
            final_risk=final_risk,
            zone=zone,
            components=components,
            confidence=confidence,
            action=action,
            steering_message=steering_message,
        )
    
    def reset_session(self, session_id: str):
        """Reset risk history for a session."""
        if session_id in self._risk_history:
            del self._risk_history[session_id]


# Quick test
if __name__ == "__main__":
    calc = RiskCalculator()
    
    test_cases = [
        ("Safe message", RiskComponents(
            lexical=0.0, semantic=0.1, short_term=0.0, long_term=0.0, adaptive=0.0
        )),
        ("Borderline educational", RiskComponents(
            lexical=0.2, semantic=0.3, short_term=0.1, long_term=0.0, adaptive=0.1
        )),
        ("Escalating session", RiskComponents(
            lexical=0.3, semantic=0.4, short_term=0.5, long_term=0.2, 
            adaptive=0.2, escalation_penalty=0.1
        )),
        ("Clear harmful", RiskComponents(
            lexical=0.9, semantic=0.8, short_term=0.0, long_term=0.0, adaptive=0.0
        )),
    ]
    
    print("Risk Calculation Tests:\n")
    for name, components in test_cases:
        result = calc.calculate(components)
        
        zone_emoji = {
            RiskZone.GREEN: "🟢",
            RiskZone.YELLOW: "🟡",
            RiskZone.RED: "🔴"
        }[result.zone]
        
        print(f"{zone_emoji} {name}")
        print(f"   Risk: {result.final_risk:.2f} ({result.zone.value})")
        print(f"   Action: {result.action}")
        print(f"   Confidence: {result.confidence:.2f}")
        if result.steering_message:
            print(f"   Message: {result.steering_message[:50]}...")
        print()
