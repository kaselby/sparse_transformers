
from enum import Enum
from typing import List


class Hook(Enum):
    IN = "IN"
    ACT = "ACT"
    UP = "UP"
    OUT = "OUT"


class ActivationCapture():
    """Helper class to capture activations from model layers."""
    hooks_available: List[Hook] = [Hook.IN, Hook.ACT, Hook.UP, Hook.OUT]
    
    def __init__(self, model):
        self.model = model
        self.mlp_activations = {
            hook: {} for hook in self.hooks_available
        }
        self.handles = []

    def _register_in_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.IN][layer_idx] = input[0].clone().detach()
            return output
        handle = layer.register_forward_hook(hook)
        return handle

    def _register_act_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.ACT][layer_idx] = output[0].clone().detach()
            return output
        handle = layer.mlp.act_fn.register_forward_hook(hook)
        return handle

    def _register_up_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.UP][layer_idx] = input[0].clone().detach()
            return output
        handle = layer.mlp.down_proj.register_forward_hook(hook)
        return handle
    
    def _register_out_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.OUT][layer_idx] = output.clone().detach()
            return output
        handle = layer.mlp.register_forward_hook(hook)
        return handle

    def get_layers(self):
        return self.model.get_decoder().layers

    def register_hooks(self, hooks):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into each transformer layer
        for i, layer in enumerate(self.get_layers()):   
            # Hooks capturing inputs to the MLP layer
            if Hook.IN in hooks and Hook.IN in self.hooks_available:
                handle = self._register_in_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)

            # Hooks capturing inputs to the activation function      
            if Hook.ACT in hooks and Hook.ACT in self.hooks_available:
                handle = self._register_act_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)

            # Hooks capturing inputs to the down projection
            if Hook.UP in hooks and Hook.UP in self.hooks_available:
                handle = self._register_up_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)

            # Hooks capturing the final MLP output
            if Hook.OUT in hooks and Hook.OUT in self.hooks_available:
                handle = self._register_out_hook(i, layer)
                if handle is not None:
                    self.handles.append(handle)

    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_captures(self):
        """Clear captured activations."""
        self.mlp_activations = {
            hook: {} for hook in self.hooks_available
        }



LOOKUP_DICT = {}

def register(*model_types):
    '''
    A decorator to record model-specific activation capture implementations.
    Arguments
    ----------
    *model_types: str
          Variable-length list of model types to register the given implementation under.
    Returns
    ----------
    callable:
          Decorator function to register model implementation.
    Examples
    ----------
    >>> @register('opt')
    ... class ActivationCaptureOPT(ActivationCapture): # This class definition gets registered for opt
    '''
    def decorator(cls):
        for model_type in model_types:
            if model_type in LOOKUP_DICT:
                raise LookupError(f"{model_type} already present") 
            LOOKUP_DICT[model_type] = cls
        return cls
    return decorator

def capture_model(model):
    '''
    Returns an ActivationCapture instance of the correct type for the given model using the 
    lookup dict (or a default implementation if no model-specific implementation is found).
    Arguments
    ----------
    model: PreTrainedModel
          Model object to use as lookup key and argument to instantiate the activation capture.
    Returns
    ----------
    ActivationCapture:
          ActivationCapture implementation stored in registry for the provided model type 
          (or default implementation as fallback if none found).
    Examples
    ----------
    >>> @register('opt')
    ... class ActivationCaptureOPT(ActivationCapture): # This class definition gets registered for opt
    >>> model.activation_capture = capture_model(model) # Loads activation capture class from registry
    >>> model.activation_capture.register_hooks()   # Use activation capture class to record model activations
    __main__.DLRM_Net
    '''
    model_type = model.model_type
    
    # Use model-specific implementation if available, otherwise fall back to default
    if model_type in LOOKUP_DICT:
        act_cls = LOOKUP_DICT[model_type]
    else:
        act_cls = ActivationCapture
    
    return act_cls(model)