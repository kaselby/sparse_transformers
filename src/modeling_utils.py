from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from transformers.modeling_outputs import (
    ModelOutput
)

@dataclass
class BaseModelOutputWithPastAndPredictorLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None    


class FastLoRAProjection(nn.Module):
    def __init__(self, hidden_size, intermediate_size, lora_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.lora_size = lora_size
        # Force creation of linear layers with actual tensors (not meta tensors)
        self.down = nn.Linear(hidden_size, lora_size, bias=False)
        self.up = nn.Linear(lora_size, intermediate_size, bias=False)
        # Pre-allocate buffers on CPU initially
        self.register_buffer('intermediate', torch.zeros(1, lora_size))
        self.register_buffer('output', torch.zeros(1, intermediate_size))
    

    def to(self, *args, **kwargs):
        # Move buffers to same device as model when .to() is called
        device = args[0] if args else kwargs.get('device')
        
        if device:
            self.intermediate = self.intermediate.to(device)
            self.output = self.output.to(device)
        return super().to(*args, **kwargs)
    
    def _resize_buffers(self, batch_size: int, dtype: torch.dtype):
        if self.intermediate.size(0) != batch_size:
            self.intermediate.resize_(batch_size, self.lora_size)
            self.intermediate = self.intermediate.to(dtype=dtype)
            self.intermediate.fill_(0.0)  # Explicitly initialize with zeros
            self.output.resize_(batch_size, self.intermediate_size)
            self.output = self.output.to(dtype=dtype)
            self.output.fill_(0.0)  # Explicitly initialize with zeros
   
    def forward(self, x):
        batch_size = x.size(0)
        
        # Check if gradients are required (training mode)
        if self.training:
            # Use regular matrix multiplication for gradient computation
            intermediate = torch.mm(x, self.down.weight.t())
            output = torch.mm(intermediate, self.up.weight.t())
            return output
        else:
            # # Use optimized in-place operations for inference
            # intermediate = torch.mm(x, self.down.weight.t())
            # output = torch.mm(intermediate, self.up.weight.t())
            # return output
        
            self._resize_buffers(batch_size, x.dtype)
            torch.mm(x, self.down.weight.t(), out=self.intermediate)
            torch.mm(self.intermediate, self.up.weight.t(), out=self.output)
            return self.output