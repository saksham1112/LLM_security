"""
ALRC v5.0 Production Pipeline
Integrated safety pipeline with Phase A (Semantic Sensor) + Phase T (Trajectory Intelligence).

Architecture:
  Phase A: Semantic Engine → Intent Profile (pure measurement)
  Phase T: Trajectory Detector → Escalation Score (temporal analysis)
  Policy: Risk Calculator → Action (uses signals from both phases)

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
from .trajectory_detector import TrajectoryDetector, build_domain_axes, TrajectorySignals
from .phase_l import LongTermMemory
from .phase_uq import UncertaintyFilter, SplitConformalPredictor, EscalationPolicy, UQLogger
from .phase_3 import (
    SafetyController, MMDDriftDetector, DriftAttributor,
    OscillationDetector, EconomicRateLimiter, GovernanceApproval,
    SafetyInvariants, SafetyWatchdog
)

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
    
    # Phase A: Intent measurement
    intent_profile: Dict[str, float] = None
    dominant_intent: str = "unknown"
    
    # Phase T: Trajectory analysis
    trajectory_signals: TrajectorySignals = None
    escalation_score: float = 0.0
    policy_state: str = "benign"
    
    # Phase L: Long-term drift
    long_term_drift: float = 0.0
    
    # Phase UQ: Uncertainty quantification
    prediction_set: set = None
    is_uncertain: bool = False
    uq_confidence: float = 1.0
    
    # Circuit breaker status
    degradation_tier: int = 1


class ALRCPipeline:
    """
    Production ALRC v5.0 Pipeline.
    
    Architecture (5 Layers):
    ┌────────────────────────────────────────────┐
    │ L1: Adaptive Normalizer (Numba)   <2ms    │
    │ L2: Session Memory (Faiss)        <5ms    │
    │ L3: Topic Memory (HNSW)           <10ms   │
    │ L4: Semantic Engine (ONNX)        <15ms   │
    │     → Phase A: Intent Profile              │
    │ L5: Trajectory Detector           <5ms    │
    │     → Phase T: Escalation Score            │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
    │ Risk Calculator (Policy)          <1ms    │
    └────────────────────────────────────────────┘
    Total: <40ms (with headroom)
    
    Circuit Breakers:
    - Tier 1: All layers (normal)
    - Tier 2: Skip trajectory (high load)
    - Tier 3: Fast path only (critical)
    """
    
    # Circuit breaker thresholds
    TIER_2_CPU_THRESHOLD = 80
    TIER_3_CPU_THRESHOLD = 95
    SEMANTIC_TIMEOUT_MS = 25
    
    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        enable_semantic: bool = True,
        enable_trajectory: bool = True,
        enable_long_term: bool = True,
        enable_uq: bool = False,
        uq_model_path: str = "models/conformal.json",
        redis_url: str = "redis://localhost:6379"
    ):
        """Initialize production pipeline."""
        logger.info("Initializing ALRC v5.0 Production Pipeline...")
        
        # Layer 1: Adaptive Normalizer (Numba JIT)
        self.normalizer = AdaptiveNormalizer()
        logger.info("  L1: Adaptive Normalizer (Numba JIT)")
        
        # Layer 2 & 3: Memory Systems (Faiss HNSW)
        self.memory = TopicMemory()
        logger.info("  L2/L3: Memory Systems (Faiss HNSW)")
        
        # Layer 4: Semantic Engine (ONNX) - Phase A
        self.enable_semantic = enable_semantic
        if enable_semantic:
            self.semantic = SemanticEngine(model_name=semantic_model)
            logger.info("  L4: Semantic Engine (ONNX) - Phase A")
        else:
            self.semantic = None
            logger.info("  L4: Semantic Engine disabled")
        
        # Layer 5: Trajectory Detector - Phase T
        self.enable_trajectory = enable_trajectory
        if enable_trajectory and enable_semantic:
            # Build domain-specific harm axes
            domain_axes = build_domain_axes(self.semantic)
            educational_protos = self.semantic._educational_prototypes
            
            self.trajectory = TrajectoryDetector(
                domain_axes=domain_axes,
                educational_prototypes=educational_protos
            )
            logger.info("  L5: Trajectory Detector - Phase T")
        else:
            self.trajectory = None
            if enable_trajectory:
                logger.warning("  L5: Trajectory disabled (requires semantic engine)")
        
        # Layer 6: Long-Term Memory - Phase L
        self.enable_long_term = enable_long_term
        if enable_long_term and enable_semantic:
            # Extract hazard centroids from semantic engine
            hazard_centroids = []
            if hasattr(self.semantic, '_harmful_centroid'):
                hazard_centroids.append(self.semantic._harmful_centroid)
            
            self.phase_l = LongTermMemory(
                redis_url=redis_url,
                hazard_centroids=hazard_centroids,
                embedding_dim=384,
                alpha=0.05,  # Prototype learning rate
                decay=0.9,   # Drift decay
                enabled=True
            )
            logger.info("  L6: Long-Term Memory - Phase L")
        else:
            self.phase_l = None
            if enable_long_term:
                logger.warning("  L6: Phase L disabled (requires semantic engine)")
        
        # Layer 7: Uncertainty Quantification - Phase UQ
        self.enable_uq = enable_uq
        if enable_uq:
            try:
                # Load calibrated predictor
                predictor = SplitConformalPredictor.load(uq_model_path)
                policy = EscalationPolicy(
                    enable_human_review=False,
                    conservative=True
                )
                self.uq_filter = UncertaintyFilter(
                    predictor=predictor,
                    escalation_policy=policy,
                    use_mvp=False,  # Start with SCP
                    enabled=True
                )
                self.uq_logger = UQLogger(log_path="logs/uq.jsonl")
                logger.info(f"  L7: Uncertainty Quantification - Phase UQ loaded from {uq_model_path}")
            except FileNotFoundError:
                logger.warning(f"  L7: Phase UQ disabled (model not found: {uq_model_path})")
                self.uq_filter = None
                self.uq_logger = None
                self.enable_uq = False
            except Exception as e:
                logger.error(f"  L7: Phase UQ initialization failed: {e}")
                self.uq_filter = None
                self.uq_logger = None
                self.enable_uq = False
        else:
            self.uq_filter = None
            self.uq_logger = None
            logger.info("  L7: Phase UQ disabled")
        
        # Risk Calculator (Policy Layer)
        self.calculator = RiskCalculator()
        logger.info("  Policy: Risk Calculator ready")
        
        # Risk Floor Manager (Topic Reset Defense)
        self.risk_floor = RiskFloorManager()
        logger.info("  Risk Floor Manager (Topic Reset Defense)")
        
        # Circuit breaker state
        self._current_tier = 1
        
        logger.info("ALRC v5.0 Production Pipeline initialized!")
    
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
        semantic_result = None
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
        
        # === Phase L: Long-Term Memory ===
        t_phase_l = time.time()
        long_term_drift = 0.0
        
        if tier == 1 and self.enable_long_term and self.phase_l and current_embedding is not None:
            try:
                # Update long-term memory and get drift signal
                long_term_drift = self.phase_l.update(
                    session_id=session_id,
                    embedding=current_embedding,
                    risk=semantic_risk,
                    intent_profile=semantic_result.intent_profile if semantic_result else {}
                )
                
                logger.debug(
                    f"Phase L: session={session_id[:8]}, "
                    f"long_term_drift={long_term_drift:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Phase L update failed: {e}")
                long_term_drift = 0.0  # Fail-safe
        
        layer_times["phase_l"] = (time.time() - t_phase_l) * 1000
        
        # === Phase T: Trajectory Analysis ===
        t_traj = time.time()
        trajectory_signals = None
        escalation_score = 0.0
        policy_state = "benign"
        
        if tier == 1 and self.enable_trajectory and self.trajectory and current_embedding is not None:
            try:
                # Track refusal (check if previous action was block/steer)
                # TODO: Track actual refusals from LLM
                refusal = False
                
                # Update trajectory and get signals
                trajectory_signals = self.trajectory.update(
                    session_id=session_id,
                    embedding=current_embedding,
                    risk=semantic_risk,
                    intent_profile=semantic_result.intent_profile if semantic_result else None,
                    refusal=refusal,
                    long_term_drift=long_term_drift  # Phase L integration
                )
                
                escalation_score = trajectory_signals.escalation_score
                policy_state = trajectory_signals.policy_state
                
                logger.debug(
                    f"Trajectory: S_t={escalation_score:.2f}, "
                    f"state={policy_state}, A_t={trajectory_signals.harm_alignment:.2f}, "
                    f"intent_slope={trajectory_signals.intent_slope:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Trajectory analysis failed: {e}")
                trajectory_signals = None
        
        layer_times["trajectory"] = (time.time() - t_traj) * 1000
        
        # === Phase UQ: Uncertainty Quantification ===
        t_uq = time.time()
        uq_decision = None
        prediction_set = None
        is_uncertain = False
        uq_confidence = 1.0
        
        if tier == 1 and self.enable_uq and self.uq_filter:
            try:
                uq_decision = self.uq_filter.filter(
                    risk_score=semantic_risk,
                    policy_state=policy_state,
                    escalation_score=escalation_score,
                    query=text,
                    long_term_drift=long_term_drift,
                    session_refusals=0  # TODO: Track actual refusals
                )
                
                prediction_set = uq_decision.prediction_set
                is_uncertain = uq_decision.set_certainty < 1.0
                uq_confidence = uq_decision.set_certainty
                
                # Log UQ decision
                if self.uq_logger:
                    self.uq_logger.log(uq_decision, {
                        'session_id': session_id,
                        'risk_score': semantic_risk,
                        'policy_state': policy_state,
                        'escalation_score': escalation_score,
                        'prompt_length': len(text.split())
                    })
                
                logger.debug(
                    f"Phase UQ: set={prediction_set}, uncertain={is_uncertain}, "
                    f"action={uq_decision.action}"
                )
                
            except Exception as e:
                logger.error(f"Phase UQ failed: {e}")
                uq_decision = None
        
        layer_times["phase_uq"] = (time.time() - t_uq) * 1000
        
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
        
        # === Phase T Policy Override ===
        # If trajectory detector determines elevated state, enhance action
        if trajectory_signals and policy_state in ["mitigate", "block"]:
            # Override action based on trajectory state
            if policy_state == "block":
                risk_result = RiskResult(
                    final_risk=max(risk_result.final_risk, 0.9),
                    zone=RiskZone.RED,
                    action="block",
                    confidence=0.9,
                    steering_message=(
                        "Conversation pattern detected. Session escalation threshold reached."
                    ),
                    components=components,
                )
            elif policy_state == "mitigate":
                risk_result = RiskResult(
                    final_risk=max(risk_result.final_risk, 0.7),
                    zone=RiskZone.YELLOW,
                    action="steer",
                    confidence=0.8,
                    steering_message=(
                        "I notice the conversation is moving toward sensitive territory. "
                        "I can explain concepts but won't provide specific implementation details."
                    ),
                    components=components,
                )
        
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
        
        # Extract Phase A intent profile
        intent_profile_dict = None
        dominant_intent = "unknown"
        if semantic_result and hasattr(semantic_result, 'intent_profile'):
            intent_profile_dict = semantic_result.intent_profile
            dominant_intent = semantic_result.dominant_intent
        
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
            # Phase A: Intent profile
            intent_profile=intent_profile_dict,
            dominant_intent=dominant_intent,
            # Phase T: Trajectory signals
            trajectory_signals=trajectory_signals,
            escalation_score=escalation_score,
            policy_state=policy_state,
            # Phase L: Long-term drift
            long_term_drift=long_term_drift,
            # Phase UQ: Uncertainty quantification
            prediction_set=prediction_set,
            is_uncertain=is_uncertain,
            uq_confidence=uq_confidence,
            degradation_tier=tier,
        )
    
    def reset_session(self, session_id: str):
        """Reset all state for a session."""
        self.memory.reset_session(session_id)
        self.calculator.reset_session(session_id)
        if self.trajectory:
            self.trajectory.reset_session(session_id)
        if self.phase_l:
            self.phase_l.reset_session(session_id)
    
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
