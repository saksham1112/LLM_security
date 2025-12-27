"""LLM interface module."""

from src.llm.base import BaseLLM, LLMResponse
from src.llm.hooks import ActivationHook, HookManager
from src.llm.ollama_backend import OllamaBackend
from src.llm.deepseek_backend import DeepSeekBackend
from src.llm.dolphin_backend import DolphinBackend, create_dolphin_backend
from src.llm.dolphin_ollama import DolphinOllamaBackend, create_dolphin_ollama_backend

__all__ = [
    "BaseLLM",
    "LLMResponse", 
    "ActivationHook",
    "HookManager",
    "OllamaBackend",
    "DeepSeekBackend",
    "DolphinBackend",
    "create_dolphin_backend",
    "DolphinOllamaBackend",
    "create_dolphin_ollama_backend",
]


