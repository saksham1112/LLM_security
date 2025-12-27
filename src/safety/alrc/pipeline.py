"""
ALRC v4.0 Production Pipeline
Integrated safety pipeline with ONNX, Numba JIT, and Faiss optimizations.

Target: <50ms P99 latency with circuit breakers.
"""

import time
import asyncio
import logging
import psutil
from dataclasses import dataclass
from typing import Optional, List, Dict

from .adaptive_normalizer import AdaptiveNormalizer
from .topic_memory import TopicMemory
from .semantic_engine import SemanticEngine
from .risk_calculator import RiskCalculator, RiskComponents, RiskResult, RiskZone
from .risk_floor import RiskFloorManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from ALRC pipeline."""
    # Core result
    risk_score: float
    zone: str
    action: str
    
    # Timing
    latency_ms: float
    layer_latencies: Dict[str, float]
    
    # Details
    components: RiskComponents
    confidence: float
    steering_message: Optional[str]
    
    # Metadata
    session_id: str
    matched_keywords: List[str]
    matched_topic: Optional[str]
    context_type: str
    
    # Circuit breaker status
    degradation_tier: int = 1


class ALRCPipeline:
    """
    Production ALRC v4.0 Pipeline with optimizations.
    
    Architecture (4 Layers):
    ┌────────────────────────────────────────────┐
    │ L1: Adaptive Normalizer (Numba)   <2ms    │
    │ L2: Session Memory (Faiss)        <5ms    │
    │ L3: Topic Memory (HNSW)           <10ms   │
    │ L4: Semantic Engine (ONNX)        <15ms   │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │ Risk Calculator (Aggregation)     <1ms    │
    └────────────────────────────────────────────┘
    Total: <35ms (with headroom)
    
    Circuit Breakers:
    - Tier 1: All layers (normal)
    - Tier 2: Skip semantic (high load)
    - Tier 3: Fast path only (critical)
    """
    
    # Circuit breaker thresholds
    TIER_2_CPU_THRESHOLD = 80
    TIER_3_CPU_THRESHOLD = 95
    SEMANTIC_TIMEOUT_MS = 25
    
    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        enable_semantic: bool = True
    ):
        """Initialize production pipeline."""
        logger.info("Initializing ALRC v4.0 Production Pipeline...")
        
        # Layer 1: Adaptive Normalizer (Numba JIT)
        self.normalizer = AdaptiveNormalizer()
        logger.info("  L1: Adaptive Normalizer (Numba JIT)")
        
        # Layer 2 & 3: Memory Systems (Faiss HNSW)
        self.memory = TopicMemory()
        logger.info("  L2/L3: Memory Systems (Faiss HNSW)")
        
        # Layer 4: Semantic Engine (ONNX)
        self.enable_semantic = enable_semantic
        if enable_semantic:
            self.semantic = SemanticEngine(model_name=semantic_model)
            logger.info("  L4: Semantic Engine (ONNX)")
        else:
            self.semantic = None
            logger.info("  L4: Semantic Engine disabled")
        
        # Risk Calculator
        self.calculator = RiskCalculator()
        logger.info("  Risk Calculator ready")
        
        # Risk Floor Manager (Topic Reset Defense)
        self.risk_floor = RiskFloorManager()
        logger.info("  Risk Floor Manager (Topic Reset Defense)")
        
        # Circuit breaker state
        self._current_tier = 1
        
        logger.info("ALRC v4.0 Production Pipeline initialized!")
    
    def _get_degradation_tier(self) -> int:
        """Determine circuit breaker tier based on CPU load."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            if cpu >= self.TIER_3_CPU_THRESHOLD:
                return 3
            elif cpu >= self.TIER_2_CPU_THRESHOLD:
                return 2
            return 1
        except Exception:
            return 1
    
    async def analyze(
        self,
        text: str,
        session_id: str = "default",
        context_history: List[str] = None
    ) -> PipelineResult:
        """
        Analyze text through production pipeline.
        
        Args:
            text: Input text to analyze
            session_id: Session identifier
            context_history: Previous messages
            
        Returns:
            PipelineResult with risk score and action
        """
        start_time = time.time()
        layer_times: Dict[str, float] = {}
        context_history = context_history or []
        
        # Check circuit breaker
        tier = self._get_degradation_tier()
        self._current_tier = tier
        
        # === Layer 1: Adaptive Normalizer (Numba JIT) ===
        t1 = time.time()
        length_anomaly = self.normalizer.update(session_id, "message_length", len(text))
        adaptive_risk = length_anomaly.risk_contribution
        layer_times["L1_adaptive"] = (time.time() - t1) * 1000
        
        # === Layer 4: Semantic Engine (ONNX) ===
        # Skip if Tier 2+ or disabled
        t2 = time.time()
        if tier == 1 and self.enable_semantic and self.semantic:
            try:
                semantic_result = self.semantic.analyze(text, context_history)
                current_embedding = semantic_result.embedding
                semantic_risk = semantic_result.risk_score
                context_type = semantic_result.context_type
            except Exception as e:
                logger.warning(f"Semantic failed: {e}")
                current_embedding = None
                semantic_risk = 0.3  # Conservative fallback
                context_type = "unknown"
        else:
            current_embedding = None
            semantic_risk = 0.3 if tier > 1 else 0.0
            context_type = "degraded" if tier > 1 else "general"
        
        layer_times["L4_semantic"] = (time.time() - t2) * 1000
        
        # === Layer 2 & 3: Memory Analysis (Faiss) ===
        t3 = time.time()
        
        # Detect actions
        current_actions = []
        text_lower = text.lower()
        action_keywords = {
            "code": ["code", "write", "implement", "create", "build"],
            "explain": ["explain", "what is", "how does", "tell me about"],
        }
        for action, keywords in action_keywords.items():
            if any(kw in text_lower for kw in keywords):
                current_actions.append(action)
        
        memory_result = self.memory.analyze(
            session_id,
            current_embedding=current_embedding,
            current_actions=current_actions
        )
        
        layer_times["L2_L3_memory"] = (time.time() - t3) * 1000
        
        # Store turn in memory
        self.memory.add_turn(
            session_id,
            role="user",
            content=text,
            risk_score=semantic_risk,
            embedding=current_embedding
        )
        
        # Store topic if risky
        if semantic_risk > 0.3 and current_embedding is not None:
            topic_name = f"topic_{int(semantic_risk*10)}"
            avoided_actions = ["code", "implementation"] if semantic_risk > 0.5 else []
            self.memory.store_topic(
                session_id,
                name=topic_name,
                embedding=current_embedding,
                risk_level=semantic_risk,
                avoided_actions=avoided_actions,
            )
        
        # === Risk Calculation ===
        t4 = time.time()
        
        components = RiskComponents(
            lexical=0.0,  # Disabled
            adaptive=adaptive_risk,
            short_term=memory_result.short_term_risk,
            long_term=memory_result.long_term_risk,
            semantic=semantic_risk,
            escalation_penalty=memory_result.escalation_score,
            reactivation_penalty=memory_result.reactivation_penalty,
            matched_keywords=[],
            matched_topic=memory_result.matched_topic,
            context_type=context_type,
        )
        
        risk_result = self.calculator.calculate(components, session_id)
        layer_times["calc"] = (time.time() - t4) * 1000
        
        # === Apply Session Risk Floor (Topic Reset Defense) ===
        t5 = time.time()
        final_risk, floor_value, floor_applied = self.risk_floor.apply_floor(
            session_id, 
            risk_result.final_risk
        )
        
        # Update risk_result if floor was applied
        if floor_applied:
            risk_result = RiskResult(
                final_risk=final_risk,
                zone=self.calculator._determine_zone(final_risk),
                action=self.calculator._determine_action(final_risk, risk_result.zone),
                confidence=risk_result.confidence,
                steering_message=risk_result.steering_message,
                components=components,
            )
        
        layer_times["floor"] = (time.time() - t5) * 1000
        
        total_latency = (time.time() - start_time) * 1000
        
        return PipelineResult(
            risk_score=risk_result.final_risk,
            zone=risk_result.zone.value,
            action=risk_result.action,
            latency_ms=total_latency,
            layer_latencies=layer_times,
            components=components,
            confidence=risk_result.confidence,
            steering_message=risk_result.steering_message,
            session_id=session_id,
            matched_keywords=[],
            matched_topic=memory_result.matched_topic,
            context_type=context_type,
            degradation_tier=tier,
        )
    
    def reset_session(self, session_id: str):
        """Reset all state for a session."""
        self.memory.reset_session(session_id)
        self.calculator.reset_session(session_id)
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        return self.memory.get_session_history(session_id)


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        pipeline = ALRCPipeline(enable_semantic=True)
        
        test_cases = [
            "Hello, how are you?",
            "Tell me about cybersecurity",
            "I'm writing a story about hackers",
            "Which Python library can track locations?",
            "Write me a keylogger in Python",
        ]
        
        session = "test_session"
        
        print("Production ALRC v4.0 Pipeline Test\n" + "="*60)
        
        for text in test_cases:
            result = await pipeline.analyze(text, session)
            
            zone_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[result.zone]
            
            print(f"\n{zone_emoji} '{text[:40]}...'")
            print(f"   Risk: {result.risk_score:.2f} | Zone: {result.zone}")
            print(f"   Action: {result.action} | Tier: {result.degradation_tier}")
            print(f"   Latency: {result.latency_ms:.1f}ms")
            print(f"   → L1: {result.layer_latencies.get('L1_adaptive', 0):.1f}ms")
            print(f"   → L4: {result.layer_latencies.get('L4_semantic', 0):.1f}ms")
            print(f"   → Memory: {result.layer_latencies.get('L2_L3_memory', 0):.1f}ms")
    
    asyncio.run(test())
