from typing import List

from src.activation_capture import ActivationCapture, Hook, register

@register('opt')
class ActivationCaptureOpt(ActivationCapture):
    hooks_available: List[Hook] = [Hook.IN, Hook.ACT, Hook.OUT]

    def _register_act_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.ACT][layer_idx] = input[0].clone().detach()
            return output
        handle = layer.activation_fn.register_forward_hook(hook)
        return handle
    
    def _register_out_hook(self, layer_idx, layer):
        def hook(module, input, output):
            # Just detach, don't clone or move to CPU yet
            self.mlp_activations[Hook.OUT][layer_idx] = output.clone().detach()
            return output
        handle = layer.fc2.register_forward_hook(hook)
        return handle

    def get_layers(self):
        return self.model.model.decoder.layers