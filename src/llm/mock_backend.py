"""
Mock LLM backend for testing.

Provides a simple mock implementation that doesn't require GPU
or model loading, useful for development and testing.
"""

import asyncio
import random
import time
from typing import Any, AsyncIterator

import torch

from src.llm.base import BaseLLM, LLMResponse


class MockLLM(BaseLLM):
    """Mock LLM for testing without GPU.
    
    Generates simple responses and simulates latency.
    Optionally simulates risky responses for testing safety systems.
    """
    
    def __init__(
        self, 
        model_name: str = "mock-llm",
        latency_ms: float = 100.0,
        simulate_risk: bool = False,
        **kwargs: Any
    ):
        """Initialize the mock LLM.
        
        Args:
            model_name: Name of the mock model.
            latency_ms: Simulated generation latency.
            simulate_risk: If True, occasionally generate risky responses.
        """
        super().__init__(model_name, **kwargs)
        self.latency_ms = latency_ms
        self.simulate_risk = simulate_risk
        self._response_templates = [
            "I understand your question about {topic}. Here's what I think...",
            "That's an interesting point about {topic}. Let me explain...",
            "Regarding {topic}, there are several perspectives to consider...",
        ]
        self._risky_templates = [
            "While I shouldn't discuss {topic} in detail, here's some information...",
            "This is sensitive, but let me explain about {topic}...",
        ]
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        params: Any = None,  # Accept dynamic control params for compatibility
    ) -> LLMResponse:
        """Generate a mock response.
        
        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens (ignored in mock).
            temperature: Sampling temperature (affects randomness).
            stop_sequences: Stop sequences (ignored in mock).
            params: Dynamic control params (used for temperature if provided).
            
        Returns:
            Mock LLMResponse.
        """
        start_time = time.time()
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Extract topic from last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        topic = last_user_msg[:50] if last_user_msg else "your question"
        
        # Select template based on risk simulation
        if self.simulate_risk and random.random() < 0.1:
            template = random.choice(self._risky_templates)
        else:
            template = random.choice(self._response_templates)
        
        text = template.format(topic=topic)
        
        # Add some variability based on temperature
        if temperature > 0.5:
            text += f" (Generated with temperature={temperature:.2f})"
        
        generation_time = (time.time() - start_time) * 1000
        
        # Generate mock hidden states if requested
        hidden_states = None
        if self._extract_hidden_states:
            # Simulate hidden states: (batch=1, seq_len, hidden_dim)
            seq_len = len(text.split())
            hidden_dim = 768
            hidden_states = torch.randn(1, seq_len, hidden_dim)
        
        return LLMResponse(
            text=text,
            tokens=[ord(c) for c in text[:100]],  # Mock tokens
            logprobs=[-0.5] * min(len(text), 100),  # Mock logprobs
            hidden_states=hidden_states,
            generation_time_ms=generation_time,
            model_name=self.model_name,
            finish_reason="stop",
        )
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream mock tokens.
        
        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens (ignored).
            temperature: Sampling temperature.
            stop_sequences: Stop sequences (ignored).
            
        Yields:
            Mock tokens.
        """
        response = await self.generate(messages, max_tokens, temperature, stop_sequences)
        words = response.text.split()
        
        for word in words:
            await asyncio.sleep(self.latency_ms / 1000 / len(words))
            yield word + " "
    
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        """Register a hook (no-op for mock).
        
        Args:
            layer_idx: Layer index.
            hook_fn: Hook function.
        """
        # Mock doesn't have real layers
        pass
    
    def remove_hooks(self) -> None:
        """Remove all hooks (no-op for mock)."""
        pass
    
    async def health_check(self) -> bool:
        """Check health (always healthy for mock).
        
        Returns:
            True (mock is always healthy).
        """
        return True
    
    async def shutdown(self) -> None:
        """Shutdown (no-op for mock)."""
        pass
