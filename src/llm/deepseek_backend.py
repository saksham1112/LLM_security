"""
DeepSeek LLM Backend.

Uses DeepSeek API (OpenAI-compatible) for chat completions.
Free tier available at: https://platform.deepseek.com/
"""

import time
from typing import Any, AsyncIterator

import httpx

from src.llm.base import BaseLLM, LLMResponse

# Import InferenceParams for dynamic control
try:
    from src.control.inference_params import InferenceParams
except ImportError:
    InferenceParams = None


class DeepSeekBackend(BaseLLM):
    """DeepSeek API backend (OpenAI-compatible).
    
    DeepSeek offers free API access with generous limits.
    API is compatible with OpenAI format.
    
    Get API key at: https://platform.deepseek.com/
    
    Example:
        backend = DeepSeekBackend(api_key="sk-...")
        response = await backend.generate([{"role": "user", "content": "Hello!"}])
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 60.0,
        **kwargs: Any
    ):
        """Initialize DeepSeek backend.
        
        Args:
            api_key: DeepSeek API key (starts with 'sk-').
            model_name: Model to use ('deepseek-chat' or 'deepseek-coder').
            base_url: API base URL.
            timeout: Request timeout in seconds.
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        params: "InferenceParams | None" = None,
    ) -> LLMResponse:
        """Generate a response from DeepSeek.
        
        Args:
            messages: Conversation history in chat format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Sequences that stop generation.
            params: Dynamic inference parameters from controller.
            
        Returns:
            LLMResponse containing generated text and metadata.
        """
        start_time = time.time()
        
        # Use dynamic params if provided (Layer 3 control)
        if params is not None:
            effective_temperature = params.temperature
        else:
            effective_temperature = temperature
        
        # Prepare request (OpenAI-compatible format)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
            "stream": False,
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
            
            # Extract response
            text = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0].get("finish_reason", "stop")
            
            generation_time = (time.time() - start_time) * 1000
            
            return LLMResponse(
                text=text,
                tokens=[],
                logprobs=[],
                hidden_states=None,
                generation_time_ms=generation_time,
                model_name=self.model_name,
                finish_reason=finish_reason,
            )
            
        except httpx.HTTPStatusError as e:
            # Try to parse error as JSON first
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", e.response.text)
                error_type = error_data.get("error", {}).get("type", "unknown")
            except:
                error_msg = e.response.text
                error_type = "unknown"
            
            raise RuntimeError(
                f"DeepSeek API error ({e.response.status_code}): {error_type} - {error_msg}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"DeepSeek request timed out after {self.timeout}s"
            ) from e
        except KeyError as e:
            raise RuntimeError(
                f"Unexpected API response format. Missing key: {e}. Response: {result if 'result' in locals() else 'No response'}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"DeepSeek generation failed: {e}") from e
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from DeepSeek.
        
        Args:
            messages: Conversation history.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            stop_sequences: Stop sequences.
            
        Yields:
            Generated tokens one at a time.
        """
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
            raise RuntimeError(f"DeepSeek streaming failed: {e}") from e
    
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        """Not supported via API."""
        pass
    
    def remove_hooks(self) -> None:
        """Not supported via API."""
        pass
    
    async def health_check(self) -> bool:
        """Check if DeepSeek API is accessible.
        
        Returns:
            True if API key is valid and API is reachable.
        """
        try:
            # Simple test request
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                }
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ DeepSeek health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "deepseek",
            "model_name": self.model_name,
            "base_url": self.base_url,
        }
