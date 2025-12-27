"""
Abstract base class for LLM backends.

This module defines the interface that all LLM backends must implement,
enabling easy swapping between vLLM, TGI, or mock implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import torch


@dataclass
class LLMResponse:
    """Response from LLM generation.
    
    Contains both the generated text and metadata for analysis.
    """
    text: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    hidden_states: torch.Tensor | None = None
    generation_time_ms: float = 0.0
    
    # Metadata
    model_name: str = ""
    finish_reason: str = "stop"  # "stop", "length", "intervention"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding tensors)."""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "logprobs": self.logprobs,
            "generation_time_ms": self.generation_time_ms,
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
        }


class BaseLLM(ABC):
    """Abstract base class for LLM backends.
    
    All LLM implementations must inherit from this class and implement
    the required methods. This enables:
    - Easy swapping between backends (vLLM, TGI, mock)
    - Consistent interface for the controller
    - Hook registration for activation extraction
    """
    
    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize the LLM backend.
        
        Args:
            model_name: Name or path of the model to load.
            **kwargs: Backend-specific configuration.
        """
        self.model_name = model_name
        self._hooks: list[Any] = []
        self._extract_hidden_states = False
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            messages: Conversation history in chat format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences that stop generation.
            
        Returns:
            LLMResponse containing generated text and metadata.
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM.
        
        Args:
            messages: Conversation history in chat format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences that stop generation.
            
        Yields:
            Generated tokens one at a time.
        """
        pass
    
    @abstractmethod
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        """Register a forward hook on a specific layer.
        
        Used for extracting hidden states for latent analysis.
        
        Args:
            layer_idx: Index of the transformer layer.
            hook_fn: Hook function to call during forward pass.
        """
        pass
    
    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        pass
    
    def enable_hidden_state_extraction(self, enabled: bool = True) -> None:
        """Enable or disable hidden state extraction.
        
        Args:
            enabled: Whether to extract hidden states during generation.
        """
        self._extract_hidden_states = enabled
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM backend is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the LLM backend."""
        pass
