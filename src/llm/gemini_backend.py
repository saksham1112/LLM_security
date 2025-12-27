"""
Gemini Backend - Model A for SxS Comparison

Uses the NEW google-genai SDK (not the legacy google-generativeai).
This is the baseline model (Google's Gemini with built-in safety).
"""

from google import genai
from dataclasses import dataclass
import time
import asyncio
import os
from typing import Optional


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    text: str
    finish_reason: str
    safety_ratings: dict
    generation_time_ms: float
    was_blocked: bool
    blocked_reason: Optional[str] = None


class GeminiProBackend:
    """
    Google Gemini backend using the new google-genai SDK.
    
    This model has Google's built-in RLHF safety.
    We use it as the baseline to compare against our Laminar system.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",  # Use latest model
    ):
        self.api_key = api_key
        self.model_name = model_name
        
        # Set API key in environment for the client
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Create client
        self.client = genai.Client(api_key=api_key)
        
        # Conversation history
        self._history = []
    
    def reset_conversation(self):
        """Reset conversation history."""
        self._history = []
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> GeminiResponse:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User message
            system_prompt: Optional system prompt
            
        Returns:
            GeminiResponse with text and metadata
        """
        start_time = time.time()
        
        try:
            # Build the full prompt with context
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"
            
            # Generate content using new SDK
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=full_prompt,
            )
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Check if blocked
            was_blocked = False
            blocked_reason = None
            response_text = ""
            
            try:
                response_text = response.text
            except Exception as e:
                # Response might be blocked
                was_blocked = True
                blocked_reason = str(e)
                response_text = f"[BLOCKED: {blocked_reason}]"
            
            # Get finish reason if available
            finish_reason = "COMPLETE"
            if hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = str(response.candidates[0].finish_reason)
            
            return GeminiResponse(
                text=response_text,
                finish_reason=finish_reason,
                safety_ratings={},
                generation_time_ms=generation_time_ms,
                was_blocked=was_blocked,
                blocked_reason=blocked_reason,
            )
            
        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            
            # Check if it's a safety block
            if "blocked" in error_str.lower() or "safety" in error_str.lower():
                return GeminiResponse(
                    text=f"[BLOCKED BY GEMINI SAFETY: {error_str}]",
                    finish_reason="SAFETY_BLOCK",
                    safety_ratings={},
                    generation_time_ms=generation_time_ms,
                    was_blocked=True,
                    blocked_reason=error_str,
                )
            else:
                return GeminiResponse(
                    text=f"[ERROR: {error_str}]",
                    finish_reason="ERROR",
                    safety_ratings={},
                    generation_time_ms=generation_time_ms,
                    was_blocked=False,
                    blocked_reason=None,
                )
    
    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents="Hello",
            )
            return True
        except:
            return False


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        backend = GeminiProBackend(
            api_key="AIzaSyC7pTwi6K2ye4WKVbCg5MszVQI5OmSj5Vw"
        )
        
        response = await backend.generate("Hello, how are you?")
        print(f"Response: {response.text}")
        print(f"Blocked: {response.was_blocked}")
    
    asyncio.run(test())
