from . import configuration_qwen_skip
from . import modelling_qwen_skip

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_mistral_skip import MistralSkipConnectionConfig
from .modelling_mistral_skip import MistralSkipConnectionForCausalLM
AutoConfig.register("mistral-skip", MistralSkipConnectionConfig)
AutoModelForCausalLM.register(MistralSkipConnectionConfig, MistralSkipConnectionForCausalLM)

__all__ = [configuration_mistral_skip, modelling_mistral_skip]