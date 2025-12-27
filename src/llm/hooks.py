"""
Activation hooks for extracting hidden states from LLM layers.

These hooks enable latent-space analysis without modifying the model.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class ActivationCapture:
    """Container for captured activations."""
    layer_idx: int
    activations: torch.Tensor
    
    @property
    def mean_activation(self) -> float:
        """Mean activation value."""
        return float(self.activations.mean())
    
    @property
    def norm(self) -> float:
        """L2 norm of activations."""
        return float(self.activations.norm())


class ActivationHook:
    """Hook for capturing activations from a specific layer.
    
    Usage:
        hook = ActivationHook(layer_idx=12)
        handle = model.layers[12].register_forward_hook(hook)
        # ... run forward pass ...
        activations = hook.get_activations()
        handle.remove()
    """
    
    def __init__(self, layer_idx: int):
        """Initialize the hook.
        
        Args:
            layer_idx: Index of the layer this hook is attached to.
        """
        self.layer_idx = layer_idx
        self._activations: torch.Tensor | None = None
        self._enabled = True
    
    def __call__(
        self, 
        module: nn.Module, 
        input: tuple[torch.Tensor, ...], 
        output: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> None:
        """Hook callback - called during forward pass.
        
        Args:
            module: The module the hook is attached to.
            input: Input to the module.
            output: Output from the module.
        """
        if not self._enabled:
            return
            
        if isinstance(output, tuple):
            # Some layers return (hidden_states, attention, ...)
            self._activations = output[0].detach().clone()
        else:
            self._activations = output.detach().clone()
    
    def get_activations(self) -> torch.Tensor | None:
        """Get captured activations."""
        return self._activations
    
    def get_capture(self) -> ActivationCapture | None:
        """Get captured activations as ActivationCapture object."""
        if self._activations is None:
            return None
        return ActivationCapture(
            layer_idx=self.layer_idx,
            activations=self._activations,
        )
    
    def clear(self) -> None:
        """Clear captured activations."""
        self._activations = None
    
    def enable(self) -> None:
        """Enable the hook."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable the hook."""
        self._enabled = False


class HookManager:
    """Manager for multiple activation hooks.
    
    Provides a convenient interface for attaching hooks to multiple layers
    and collecting all activations at once.
    """
    
    def __init__(self):
        """Initialize the hook manager."""
        self._hooks: dict[int, ActivationHook] = {}
        self._handles: list[Any] = []
    
    def attach_to_model(
        self, 
        model: nn.Module, 
        layer_indices: list[int],
        layer_accessor: Callable[[nn.Module, int], nn.Module] | None = None
    ) -> None:
        """Attach hooks to specified layers of a model.
        
        Args:
            model: The model to attach hooks to.
            layer_indices: Indices of layers to hook.
            layer_accessor: Function to get layer from model by index.
                           Default assumes model.layers[idx] or model.model.layers[idx].
        """
        if layer_accessor is None:
            # Default accessor for common architectures
            def layer_accessor(m: nn.Module, idx: int) -> nn.Module:
                if hasattr(m, "layers"):
                    return m.layers[idx]
                elif hasattr(m, "model") and hasattr(m.model, "layers"):
                    return m.model.layers[idx]
                else:
                    raise ValueError(
                        "Could not find layers in model. "
                        "Provide a custom layer_accessor."
                    )
        
        for idx in layer_indices:
            hook = ActivationHook(layer_idx=idx)
            layer = layer_accessor(model, idx)
            handle = layer.register_forward_hook(hook)
            self._hooks[idx] = hook
            self._handles.append(handle)
    
    def get_all_activations(self) -> dict[int, torch.Tensor]:
        """Get activations from all hooked layers.
        
        Returns:
            Dictionary mapping layer index to activations tensor.
        """
        result = {}
        for idx, hook in self._hooks.items():
            act = hook.get_activations()
            if act is not None:
                result[idx] = act
        return result
    
    def get_activation_norms(self) -> dict[int, float]:
        """Get L2 norms of activations from all hooked layers.
        
        Returns:
            Dictionary mapping layer index to activation norm.
        """
        result = {}
        for idx, hook in self._hooks.items():
            capture = hook.get_capture()
            if capture is not None:
                result[idx] = capture.norm
        return result
    
    def clear_all(self) -> None:
        """Clear all captured activations."""
        for hook in self._hooks.values():
            hook.clear()
    
    def remove_all(self) -> None:
        """Remove all hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._hooks.clear()
        self._handles.clear()
    
    def enable_all(self) -> None:
        """Enable all hooks."""
        for hook in self._hooks.values():
            hook.enable()
    
    def disable_all(self) -> None:
        """Disable all hooks."""
        for hook in self._hooks.values():
            hook.disable()
