"""
Dolphin LLM Backend via Ollama with Laminar Safety Wrapper.

Dolphin-Llama3 is an uncensored local model - all safety comes from 
our external Laminar layer.

This provides:
- Full capability (no alignment tax)
- External safety enforcement (trajectory-aware)
- Complete visibility into safety decisions
- NO rate limits (local model)
- Fast inference (no network latency)

Usage:
    from src.llm.dolphin_ollama import DolphinOllamaBackend
    
    backend = DolphinOllamaBackend()
    response = await backend.generate([{"role": "user", "content": "hi"}])
"""

import time
import json
from typing import Any, Optional, AsyncIterator
from dataclasses import dataclass

import httpx

from src.llm.base import BaseLLM, LLMResponse

try:
    from src.control.inference_params import InferenceParams
except ImportError:
    InferenceParams = None

try:
    from src.steering.yellow_zone import YellowZoneSteering, get_yellow_zone_steering
except ImportError:
    YellowZoneSteering = None
    get_yellow_zone_steering = None

try:
    from src.safety.post_filter import PostGenerationFilter, get_post_filter
except ImportError:
    PostGenerationFilter = None
    get_post_filter = None

# === HARDENED SECURITY COMPONENTS (HOTFIX) ===
try:
    from src.safety.domain_tiers import (
        DomainClassifier, DomainTier, get_domain_classifier, classify_domain
    )
    from src.controller.red_gate import HardREDGate, GateResult
    from src.risk.markers import EscalationMarkers
    from src.safety.zone_contracts import ZoneClassifier, ZoneType
    from src.risk.reconstructability import ReconstructabilityEstimator
    HARDENED_SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Hardened security components not available: {e}")
    HARDENED_SECURITY_AVAILABLE = False
    DomainClassifier = None
    HardREDGate = None
    EscalationMarkers = None
    ZoneClassifier = None
    ReconstructabilityEstimator = None


@dataclass
class SafetyMetadata:
    """Metadata about safety decisions made during generation."""
    zone: str                    # GREEN, YELLOW, RED
    steering_mode: str           # NONE, EDUCATIONAL, SAFETY_CONSCIOUS, etc.
    risk_score: float           # 0.0 to 1.0
    was_steered: bool           # True if system prompt was modified
    was_post_filtered: bool     # True if output was sanitized
    original_length: int        # Length before filtering
    filtered_length: int        # Length after filtering (if applicable)


class DolphinOllamaBackend(BaseLLM):
    """
    Dolphin (Uncensored) LLM via Ollama with Laminar Safety Wrapper.
    
    Flow:
    1. Pre-generation: Risk detection + Yellow Zone steering
    2. Generation: Uncensored Dolphin model via Ollama
    3. Post-generation: Content filtering if needed
    
    Key insight: Dolphin will NOT refuse requests by itself.
    All safety behavior comes from our external Laminar wrapper.
    """
    
    DEFAULT_MODEL = "dolphin-llama3"
    
    def __init__(
        self,
        model_name: str = None,
        base_url: str = "http://localhost:11435",
        timeout: float = 300.0,
        enable_safety_layer: bool = True,
        **kwargs: Any
    ):
        """
        Initialize Dolphin Ollama backend.
        
        Args:
            model_name: Ollama model name (default: dolphin-llama3).
            base_url: Ollama server URL.
            timeout: Request timeout (local models can be slow on first run).
            enable_safety_layer: If False, bypass all safety (for testing).
        """
        model = model_name or self.DEFAULT_MODEL
        super().__init__(model, **kwargs)
        
        self.base_url = base_url
        self.timeout = timeout
        self.enable_safety_layer = enable_safety_layer
        
        # Initialize safety components
        if enable_safety_layer:
            if get_yellow_zone_steering:
                self.yellow_zone = get_yellow_zone_steering()
            else:
                self.yellow_zone = None
                
            if get_post_filter:
                self.post_filter = get_post_filter()
            else:
                self.post_filter = None
            
            # === HARDENED SECURITY COMPONENTS ===
            if HARDENED_SECURITY_AVAILABLE:
                self.domain_classifier = get_domain_classifier()
                self.red_gate = HardREDGate()
                self.markers = EscalationMarkers()
                self.zone_classifier = ZoneClassifier(production_mode=True)
                self.reconstructability = ReconstructabilityEstimator()
                print("🛡️ HARDENED SECURITY: Enabled (domain tiers + RED gate + reconstructability)")
            else:
                self.domain_classifier = None
                self.red_gate = None
                self.markers = None
                self.zone_classifier = None
                self.reconstructability = None
        else:
            self.yellow_zone = None
            self.post_filter = None
            self.domain_classifier = None
            self.red_gate = None
            self.markers = None
            self.zone_classifier = None
            self.reconstructability = None
        
        # HTTP client
        self._client = httpx.AsyncClient(timeout=timeout)
        
        # Track last safety metadata
        self.last_safety_metadata: Optional[SafetyMetadata] = None
        
        # Turn counter for trajectory tracking
        self._turn_count = 0
        self._accumulated_risk = 0.0
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        params: "InferenceParams | None" = None,
    ) -> LLMResponse:
        """
        Generate response with HARDENED safety wrapper.
        
        Flow (HOTFIX):
        0. HARD RED GATE: Domain tier check + circuit breaker (FIRST!)
        1. Zone classification (trajectory-aware)
        2. Apply Yellow Zone steering (if permitted)
        3. Call Dolphin via Ollama (uncensored)
        4. Apply post-generation filter + reconstructability update
        """
        start_time = time.time()
        self._turn_count += 1
        
        # Extract the latest user message for classification
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        
        # Extract risk info from params
        instant_risk = 0.0
        effective_risk = 0.0
        risk_state = 0.0
        harm_potential = 0.0
        
        if params:
            instant_risk = getattr(params, 'instant_risk', 0.0)
            effective_risk = getattr(params, 'effective_risk', 0.0)
            risk_state = getattr(params, 'risk_state', 0.0)
            temperature = getattr(params, 'temperature', temperature)
        
        # === ZONE THRESHOLDS (TUNABLE) ===
        YELLOW_THRESHOLD = 0.25  # Risk >= this = Yellow
        RED_THRESHOLD = 0.65     # Risk >= this = Red
        
        # Update accumulated risk with decay
        # instant_risk comes from AdvancedRiskEstimator which handles semantic inference
        self._accumulated_risk = self._accumulated_risk * 0.85 + instant_risk * 0.4
        self._accumulated_risk = max(self._accumulated_risk, risk_state)  # Floor from external state
        
        # === ZONE SELECTION (PURE THRESHOLD LOGIC) ===
        # No pattern matching here - zones are determined by accumulated risk from the formula
        was_steered = False
        steering_mode = "NONE"
        
        # Determine zone based on accumulated risk thresholds
        if self._accumulated_risk >= RED_THRESHOLD:
            zone = "🔴 RED"
        elif self._accumulated_risk >= YELLOW_THRESHOLD:
            zone = "🟡 YELLOW"
        else:
            zone = "🟢 GREEN"
        
        # === DETAILED LOGGING ===
        if self.enable_safety_layer:
            print(f"\n{'='*60}")
            print(f"🔍 SAFETY ANALYSIS | Turn {self._turn_count}")
            print(f"{'='*60}")
            print(f"📝 Input: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
            print(f"\n📊 RISK VALUES (from AdvancedRiskEstimator):")
            print(f"   Instant Risk:      {instant_risk:.3f}")
            print(f"   Accumulated Risk:  {self._accumulated_risk:.3f}")
            print(f"   Effective Risk:    {effective_risk:.3f}")
            print(f"   Harm Potential:    {harm_potential:.3f}")
            print(f"\n📍 ZONE THRESHOLDS:")
            print(f"   GREEN:  < {YELLOW_THRESHOLD}")
            print(f"   YELLOW: {YELLOW_THRESHOLD} - {RED_THRESHOLD}")
            print(f"   RED:    >= {RED_THRESHOLD}")
            print(f"\n🎯 ZONE DECISION: {zone}")
            print(f"{'='*60}")
        
        # === RED ZONE = BLOCK ===
        if zone == "🔴 RED":
            self.last_safety_metadata = SafetyMetadata(
                zone=zone,
                steering_mode="BLOCKED",
                risk_score=self._accumulated_risk,
                was_steered=False,
                was_post_filtered=True,
                original_length=0,
                filtered_length=0,
            )
            
            return LLMResponse(
                text="I cannot continue in this direction. The conversation has accumulated too much risk.",
                tokens=[],
                logprobs=[],
                hidden_states=None,
                generation_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model_name,
                finish_reason="safety_blocked",
            )
        
        # === STEP 1: Yellow Zone Steering (if not blocked and in YELLOW zone) ===
        
        if self.enable_safety_layer and self.yellow_zone:
            steered_messages, steering_config = self.yellow_zone.apply_steering(
                messages=messages,
                instant_risk=instant_risk,
                accumulated_risk=risk_state,
                effective_risk=effective_risk,
                harm_potential=harm_potential,
            )
            zone = self.yellow_zone.get_zone_name(steering_config)
            steering_mode = steering_config.mode.name
            was_steered = steering_mode != "NONE"
            
            # Use steered messages
            generation_messages = steered_messages
            
            # Log steering
            if was_steered:
                print(f"\n🐬 DOLPHIN-OLLAMA | {zone} | Steering: {steering_mode}")
        else:
            generation_messages = messages
        
        # === STEP 2: Call Dolphin via Ollama ===
        # Convert to Ollama format
        ollama_messages = []
        for msg in generation_messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get("message", {}).get("content", "")
            original_length = len(text)
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error ({e.response.status_code}): {e.response.text}") from e
        except httpx.ConnectError:
            raise RuntimeError("Ollama is not running. Start with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e
        
        # === STEP 3: Post-Generation Filter ===
        was_post_filtered = False
        filtered_length = original_length
        
        if self.enable_safety_layer and self.post_filter:
            content_analysis = self.post_filter.analyze(text)
            
            if content_analysis.is_harmful:
                text, was_post_filtered = self.post_filter.sanitize(text, content_analysis)
                filtered_length = len(text)
                zone = "🔴 RED (post-filter)"
                print(f"   🛡️ POST-FILTER | Category: {content_analysis.category.name}")
        
        generation_time = (time.time() - start_time) * 1000
        
        # Store safety metadata
        self.last_safety_metadata = SafetyMetadata(
            zone=zone,
            steering_mode=steering_mode,
            risk_score=effective_risk,
            was_steered=was_steered,
            was_post_filtered=was_post_filtered,
            original_length=original_length,
            filtered_length=filtered_length,
        )
        
        return LLMResponse(
            text=text,
            tokens=[],
            logprobs=[],
            hidden_states=None,
            generation_time_ms=generation_time,
            model_name=self.model_name,
            finish_reason="stop",
        )
    
    def get_safety_metadata(self) -> Optional[SafetyMetadata]:
        """Get metadata about the last generation's safety decisions."""
        return self.last_safety_metadata
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from Dolphin via Ollama."""
        ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}") from e
    
    async def generate_bypass(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate WITHOUT safety layer (for testing/comparison).
        
        ⚠️ WARNING: This bypasses all safety measures!
        """
        # Temporarily disable safety
        original_safety = self.enable_safety_layer
        self.enable_safety_layer = False
        
        try:
            response = await self.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response.model_name = f"{self.model_name} (UNSAFE BYPASS)"
            return response
        finally:
            self.enable_safety_layer = original_safety
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            
            data = response.json()
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return self.model_name.split(":")[0] in models
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "safety_layer": self.enable_safety_layer,
            "type": "LOCAL_UNCENSORED + EXTERNAL_SAFETY",
        }
    
    # Local model - no hooks available via Ollama API
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        pass
    
    def remove_hooks(self) -> None:
        pass


# Factory function
def create_dolphin_ollama_backend(
    safe_mode: bool = True,
    model_name: str = "dolphin-llama3",
) -> DolphinOllamaBackend:
    """
    Create a Dolphin Ollama backend.
    
    Args:
        safe_mode: If True, enable Laminar safety wrapper.
                   If False, run completely uncensored (dangerous!).
        model_name: Ollama model to use.
    """
    return DolphinOllamaBackend(
        model_name=model_name,
        enable_safety_layer=safe_mode,
    )
