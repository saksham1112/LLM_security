"""
Ollama backend for LLM inference.

Connects to local Ollama instance running models like Llama 3.
Ollama provides a simple REST API for text generation.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx
import torch

from src.llm.base import BaseLLM, LLMResponse

# Import InferenceParams for dynamic control
try:
    from src.control.inference_params import InferenceParams
except ImportError:
    InferenceParams = None  # Type stub for optional dependency


class OllamaBackend(BaseLLM):
    """Ollama backend for local LLM inference.
    
    Connects to Ollama API (default: http://localhost:11434)
    Supports any model available in Ollama (llama3, mistral, etc.)
    
    Example:
        backend = OllamaBackend(model_name="llama3")
        response = await backend.generate([{"role": "user", "content": "Hello!"}])
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any
    ):
        """Initialize the Ollama backend.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama3", "mistral").
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
            **kwargs: Additional configuration.
        """
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        params: "InferenceParams | None" = None,  # Dynamic control params
    ) -> LLMResponse:
        """Generate a response from Ollama.
        
        Args:
            messages: Conversation history in chat format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences that stop generation.
            
        Returns:
            LLMResponse containing generated text and metadata.
        """
        start_time = time.time()
        
        # Use dynamic params if provided (Layer 3 control)
        if params is not None:
            effective_temperature = params.temperature
        else:
            effective_temperature = temperature
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": effective_temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            # Call Ollama API
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response
            text = result.get("message", {}).get("content", "")
            
            generation_time = (time.time() - start_time) * 1000
            
            # Extract hidden states if requested (Ollama doesn't provide this easily)
            hidden_states = None
            if self._extract_hidden_states:
                # Would need to use Ollama's embeddings API or model internals
                # For now, we'll skip this - can add later if needed
                pass
            
            return LLMResponse(
                text=text,
                tokens=[],  # Ollama doesn't return tokens in this format
                logprobs=[],  # Ollama doesn't return logprobs easily
                hidden_states=hidden_states,
                generation_time_ms=generation_time,
                model_name=self.model_name,
                finish_reason=result.get("done_reason", "stop"),
            )
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama API error (status {e.response.status_code}): {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama.
        
        Args:
            messages: Conversation history in chat format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences that stop generation.
            
        Yields:
            Generated tokens one at a time.
        """
        # Prepare request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            if content:
                                yield content
                        
                        # Check if done
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama streaming error (status {e.response.status_code})"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}") from e
    
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        """Register a forward hook (not supported by Ollama API).
        
        Ollama doesn't expose model internals via API.
        For latent analysis, you'd need to run the model locally with transformers.
        
        Args:
            layer_idx: Layer index.
            hook_fn: Hook function.
        """
        # Not supported via Ollama API
        # Could potentially be added if we switch to using Ollama's model
        # files directly with transformers
        pass
    
    def remove_hooks(self) -> None:
        """Remove all hooks (no-op for Ollama)."""
        pass
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available.
        
        Returns:
            True if Ollama is healthy and model exists.
        """
        try:
            # Check if Ollama is running
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Check if our model is pulled
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            # Ollama sometimes adds :latest tag
            model_exists = any(
                self.model_name in name for name in model_names
            )
            
            if not model_exists:
                print(f"⚠️  Model '{self.model_name}' not found in Ollama.")
                print(f"   Available models: {model_names}")
                print(f"   Run: ollama pull {self.model_name}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Ollama health check failed: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the client."""
        await self._client.aclose()
    
    async def pull_model(self) -> bool:
        """Pull the model if not already available.
        
        Returns:
            True if model is now available.
        """
        try:
            print(f"📥 Pulling model '{self.model_name}' from Ollama...")
            
            response = await self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name, "stream": False},
                timeout=600.0,  # 10 minutes for model download
            )
            response.raise_for_status()
            
            print(f"✅ Model '{self.model_name}' pulled successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to pull model: {e}")
            return False
    
    def get_info(self) -> dict[str, Any]:
        """Get backend information.
        
        Returns:
            Dictionary with backend info.
        """
        return {
            "backend": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
