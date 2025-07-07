from typing_extensions import override
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ActivationCapture(ABC):
    """Helper class to capture activations from model layers."""    
    def __init__(self, model):
        self.model = model
        self.mlp_activations = {}
        self.handles = []
        

    @abstractmethod
    def _register_activation_hook(self, layer_idx, layer):
        pass

    @abstractmethod
    def get_layers(self):
        pass

    @abstractmethod
    def get_mlp_activations(self, layer_idx):
        """Get combined MLP activations for a layer."""
        pass

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers()):
            handle = self._register_activation_hook(i, layer)   
            if handle is not None:
                self.handles.append(handle)    
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_captures(self):
        """Clear captured activations."""
        self.mlp_activations = {}



class ActivationCaptureDefault(ActivationCapture):
    """Helper class to capture activations from model layers."""
    
    def get_layers(self):
        return self.model.get_decoder().layers

    def _create_mlp_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            key = f"{layer_idx}_{proj_type}"
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[key] = output.clone().detach()
            return output
        return hook

    def _register_activation_hook(self, layer_idx, layer):
        handle = layer.mlp.act_fn.register_forward_hook(
            self._create_mlp_hook(layer_idx, 'act')
        )
        return handle
    
    def get_mlp_activations(self, layer_idx):
        act_key = f"{layer_idx}act"
        if act_key in self.mlp_activations:
            act = self.mlp_activations[act_key]
            return act
        return None


class ActivationCaptureTraining(ActivationCaptureDefault):
    """Additional Hidden State capture for training dataset generation"""
    def __init__(self, model):
        super().__init__(model)
        self.hidden_states = {}
    
    def _create_hidden_state_hook(self, layer_idx, layer):
        def hook(module, args, kwargs, output):
            # args[0] is the input hidden states to the layer
            if len(args) > 0:
                # Just detach, don't clone or move to CPU yet
                self.hidden_states[layer_idx] = args[0].clone().detach()
            return output
        return hook
    
    def _register_hidden_state_hook(self, layer_idx, layer):
        handle = layer.register_forward_hook(
            self._create_hidden_state_hook(layer_idx, layer),
            with_kwargs=True
        )
        return handle

    @override
    def clear_captures(self):
        """Clear captured activations."""
        super().clear_captures()
        self.hidden_states = {}

    @override
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        super().register_hooks()
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers()):            
            # Capture hidden states before MLP
            handle = self._register_hidden_state_hook(i, layer)
            if handle is not None:
                self.handles.append(handle)
    
    def get_hidden_states(self, layer_idx):
        """Get hidden states for a layer."""
        return self.hidden_states[layer_idx]
