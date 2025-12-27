"""
TinyLlama Backend for Response Generation + Safety Probe

Uses TinyLlama-1.1B-Chat from HuggingFace for BOTH:
1. Response generation (chat-quality output)
2. Hidden state extraction (for safety probe)

This is Model B in the SxS comparison - our Laminar-protected system.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional
import time
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class PythiaResponse:
    """Response from TinyLlama backend."""
    text: str
    generation_time_ms: float
    finish_reason: str
    zone: str
    risk_score: float
    was_blocked: bool
    metadata: dict


class PythiaBackend:
    """
    HuggingFace TinyLlama-1.1B-Chat backend with integrated safety probe.
    
    Combines:
    - Text generation (chat-quality response to user)
    - Hidden state extraction (for risk calculation)
    - Zone-based safety control (GREEN/YELLOW/RED)
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        probe_path: str = "models/safety_probe.pt",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.probe_path = probe_path
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.probe = None
        
        self._loaded = False
        self._accumulated_risk = 0.0
        
        # Zone thresholds
        self.YELLOW_THRESHOLD = 0.25
        self.RED_THRESHOLD = 0.65
        
        # TinyLlama has 22 layers - use middle layers for probe
        self.probe_layers = [8, 9, 10, 11, 12, 13, 14, 15]
    
    def load(self) -> bool:
        """Load model, tokenizer, and probe."""
        if self._loaded:
            return True
        
        try:
            logger.info(f"Loading Pythia model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with hidden states
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                output_hidden_states=True,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load probe
            try:
                from src.probe.linear_probe import LinearProbe
                hidden_dim = self.model.config.hidden_size
                self.probe = LinearProbe(input_dim=hidden_dim * len(self.probe_layers))
                self.probe.load_state_dict(torch.load(self.probe_path))
                self.probe.eval()
                logger.info(f"Loaded probe from {self.probe_path}")
            except Exception as e:
                logger.warning(f"Could not load probe: {e}. Using untrained probe.")
                from src.probe.linear_probe import LinearProbe
                self.probe = LinearProbe(input_dim=self.model.config.hidden_size * len(self.probe_layers))
            
            self._loaded = True
            logger.info(f"Pythia backend ready on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Pythia: {e}")
            return False
    
    @torch.no_grad()
    def _extract_risk(self, input_ids: torch.Tensor, hidden_states: tuple) -> float:
        """Extract risk score from hidden states using probe."""
        try:
            layer_features = []
            for idx in self.probe_layers:
                if idx < len(hidden_states):
                    hidden = hidden_states[idx]
                    # Mean pool across tokens
                    pooled = hidden.mean(dim=1)
                    layer_features.append(pooled)
            
            if not layer_features:
                return 0.0
            
            features = torch.cat(layer_features, dim=-1)
            risk = self.probe(features).item()
            return risk
        except Exception as e:
            logger.warning(f"Risk extraction failed: {e}")
            return 0.0
    
    def _get_zone(self, accumulated_risk: float) -> str:
        """Determine zone from accumulated risk."""
        if accumulated_risk >= self.RED_THRESHOLD:
            return "🔴 RED"
        elif accumulated_risk >= self.YELLOW_THRESHOLD:
            return "🟡 YELLOW"
        else:
            return "🟢 GREEN"
    
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        context: Optional[list[str]] = None,
    ) -> PythiaResponse:
        """
        Generate response with integrated safety check.
        
        Args:
            prompt: User message
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            context: Previous conversation context
            
        Returns:
            PythiaResponse with text, risk info, and zone
        """
        # Lazy load
        if not self._loaded:
            await asyncio.to_thread(self.load)
        
        start_time = time.time()
        
        # Use TinyLlama's chat template for proper chat responses
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass to get hidden states for risk
        outputs = self.model(**inputs, output_hidden_states=True)
        instant_risk = self._extract_risk(inputs["input_ids"], outputs.hidden_states)
        
        # Update accumulated risk
        self._accumulated_risk = self._accumulated_risk * 0.85 + instant_risk * 0.4
        self._accumulated_risk = max(self._accumulated_risk, instant_risk)
        
        zone = self._get_zone(self._accumulated_risk)
        
        # Check if blocked
        if self._accumulated_risk >= self.RED_THRESHOLD:
            generation_time_ms = (time.time() - start_time) * 1000
            return PythiaResponse(
                text="[BLOCKED: Risk threshold exceeded. Cannot continue in this direction.]",
                generation_time_ms=generation_time_ms,
                finish_reason="safety_blocked",
                zone=zone,
                risk_score=self._accumulated_risk,
                was_blocked=True,
                metadata={
                    "instant_risk": instant_risk,
                    "accumulated_risk": self._accumulated_risk,
                    "zone": zone,
                },
            )
        
        # Generate response
        try:
            gen_outputs = await asyncio.to_thread(
                self.model.generate,
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Decode response (exclude input)
            response_ids = gen_outputs[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
        except Exception as e:
            response_text = f"[Generation error: {e}]"
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        return PythiaResponse(
            text=response_text,
            generation_time_ms=generation_time_ms,
            finish_reason="complete",
            zone=zone,
            risk_score=self._accumulated_risk,
            was_blocked=False,
            metadata={
                "instant_risk": instant_risk,
                "accumulated_risk": self._accumulated_risk,
                "zone": zone,
            },
        )
    
    def reset_conversation(self):
        """Reset accumulated risk for new conversation."""
        self._accumulated_risk = 0.0
    
    async def health_check(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        backend = PythiaBackend()
        
        response = await backend.generate("What is the capital of France?")
        print(f"Response: {response.text}")
        print(f"Zone: {response.zone}")
        print(f"Risk: {response.risk_score:.3f}")
    
    asyncio.run(test())
