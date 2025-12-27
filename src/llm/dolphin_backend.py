"""
Dolphin LLM Backend with Laminar Safety Wrapper.

Dolphin is an "uncensored" model from Cognitive Computations.
It has no built-in safety refusals - all safety comes from OUR external layer.

This is the "Glass Box" approach:
- Full capability (no alignment tax)
- External safety enforcement (trajectory-aware)
- Complete visibility into safety decisions

Available models on OpenRouter:
- cognitivecomputations/dolphin-mistral-24b-venice-edition:free (FREE!)
- cognitivecomputations/dolphin3.0-r1-mistral-24b:free
- cognitivecomputations/dolphin-2.9-llama3-8b
"""

import time
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


class DolphinBackend(BaseLLM):
    """
    Dolphin (Uncensored) LLM with Laminar Safety Wrapper.
    
    This backend wraps an uncensored model with our external safety layer:
    1. Pre-generation: Risk detection + Yellow Zone steering
    2. Generation: Uncensored Dolphin model
    3. Post-generation: Content filtering if needed
    
    The key insight: Dolphin will NOT refuse requests by itself.
    All safety behavior comes from our external Laminar wrapper.
    """
    
    # Default to free Venice edition
    DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
    
    def __init__(
        self,
        api_key: str,
        model_name: str = None,
        timeout: float = 90.0,
        enable_safety_layer: bool = True,
        **kwargs: Any
    ):
        """
        Initialize Dolphin backend.
        
        Args:
            api_key: OpenRouter API key.
            model_name: Dolphin model variant to use.
            timeout: Request timeout (Dolphin can be slower).
            enable_safety_layer: If False, bypass all safety (for testing).
        """
        model = model_name or self.DEFAULT_MODEL
        super().__init__(model, **kwargs)
        
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
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
        else:
            self.yellow_zone = None
            self.post_filter = None
        
        # HTTP client
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Laminar Safety Evaluation Platform",
            }
        )
        
        # Track last safety metadata
        self.last_safety_metadata: Optional[SafetyMetadata] = None
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        params: "InferenceParams | None" = None,
    ) -> LLMResponse:
        """
        Generate response with safety wrapper.
        
        Flow:
        1. Apply Yellow Zone steering (if risk is moderate)
        2. Call Dolphin (uncensored)
        3. Apply post-generation filter (if needed)
        """
        start_time = time.time()
        
        # Extract risk info from params
        instant_risk = 0.0
        effective_risk = 0.0
        harm_potential = 0.0
        
        if params:
            instant_risk = getattr(params, 'instant_risk', 0.0)
            effective_risk = getattr(params, 'effective_risk', 0.0)
            risk_state = getattr(params, 'risk_state', 0.0)
            temperature = getattr(params, 'temperature', temperature)
        
        # === STEP 1: Yellow Zone Steering ===
        was_steered = False
        steering_mode = "NONE"
        zone = "GREEN"
        
        if self.enable_safety_layer and self.yellow_zone:
            steered_messages, steering_config = self.yellow_zone.apply_steering(
                messages=messages,
                instant_risk=instant_risk,
                accumulated_risk=risk_state if params else 0.0,
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
                print(f"\n🐬 DOLPHIN | {zone} | Steering: {steering_mode}")
        else:
            generation_messages = messages
        
        # === STEP 2: Call Dolphin (uncensored) ===
        payload = {
            "model": self.model_name,
            "messages": generation_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason", "stop")
            original_length = len(text)
            
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", e.response.text)
            except:
                error_msg = e.response.text
            raise RuntimeError(f"Dolphin API error ({e.response.status_code}): {error_msg}") from e
        
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
            finish_reason=finish_reason,
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
        """Stream tokens from Dolphin (without safety filtering during stream)."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        import json
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise RuntimeError(f"Dolphin streaming failed: {e}") from e
    
    async def generate_bypass(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate WITHOUT safety layer (for testing/comparison).
        
        ⚠️ WARNING: This bypasses all safety measures!
        Use only for validating that Dolphin is truly uncensored.
        """
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        generation_time = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=text,
            tokens=[],
            logprobs=[],
            hidden_states=None,
            generation_time_ms=generation_time,
            model_name=f"{self.model_name} (UNSAFE BYPASS)",
            finish_reason="stop",
        )
    
    async def health_check(self) -> bool:
        """Check if Dolphin model is accessible."""
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                }
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "dolphin",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "safety_layer": self.enable_safety_layer,
            "type": "UNCENSORED + EXTERNAL_SAFETY",
        }
    
    # API-only backend - no hooks available
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        pass
    
    def remove_hooks(self) -> None:
        pass


# Factory function
def create_dolphin_backend(
    api_key: str,
    safe_mode: bool = True,
) -> DolphinBackend:
    """
    Create a Dolphin backend.
    
    Args:
        api_key: OpenRouter API key.
        safe_mode: If True, enable Laminar safety wrapper.
                   If False, run completely uncensored (dangerous!).
    """
    return DolphinBackend(
        api_key=api_key,
        enable_safety_layer=safe_mode,
    )
