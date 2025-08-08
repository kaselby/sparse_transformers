from . import configuration_opt_skip
from . import modelling_opt_skip
from . import activation_capture_opt

from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_opt_skip import OPTSkipConnectionConfig
from .modelling_opt_skip import OPTSkipConnectionForCausalLM
AutoConfig.register("opt-skip", OPTSkipConnectionConfig)
AutoModelForCausalLM.register(OPTSkipConnectionConfig, OPTSkipConnectionForCausalLM)

__all__ = [configuration_opt_skip, modelling_opt_skip]