"""
OpenRouter LLM Backend.

OpenRouter provides access to many LLMs through a unified API.
API is OpenAI-compatible.
"""

import time
from typing import Any, AsyncIterator

import httpx

from src.llm.base import BaseLLM, LLMResponse

try:
    from src.control.inference_params import InferenceParams
except ImportError:
    InferenceParams = None


class OpenRouterBackend(BaseLLM):
    """OpenRouter API backend (OpenAI-compatible).
    
    OpenRouter provides access to many models:
    - deepseek/deepseek-chat
    - meta-llama/llama-3-8b-instruct
    - google/gemma-7b-it
    - And many more
    
    Get API key at: https://openrouter.ai/
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek/deepseek-chat",
        timeout: float = 60.0,
        **kwargs: Any
    ):
        """Initialize OpenRouter backend.
        
        Args:
            api_key: OpenRouter API key (starts with 'sk-or-v1-').
            model_name: Model to use (e.g., 'deepseek/deepseek-chat').
            timeout: Request timeout in seconds.
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "LLM Safety Control Dashboard",
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
        """Generate a response from OpenRouter."""
        start_time = time.time()
        
        # Use dynamic params if provided
        if params is not None:
            effective_temperature = params.temperature
        else:
            effective_temperature = temperature
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": effective_temperature,
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
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", e.response.text)
            except:
                error_msg = e.response.text
            raise RuntimeError(f"OpenRouter API error ({e.response.status_code}): {error_msg}") from e
        except Exception as e:
            raise RuntimeError(f"OpenRouter generation failed: {e}") from e
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from OpenRouter."""
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
            raise RuntimeError(f"OpenRouter streaming failed: {e}") from e
    
    def register_hook(self, layer_idx: int, hook_fn: Any) -> None:
        """Not supported via API."""
        pass
    
    def remove_hooks(self) -> None:
        """Not supported via API."""
        pass
    
    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    def get_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            "backend": "openrouter",
            "model_name": self.model_name,
            "base_url": self.base_url,
        }
