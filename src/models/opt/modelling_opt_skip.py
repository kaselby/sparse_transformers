import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import dataclass
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast
)
from transformers.models.opt.modeling_opt import(
     OPTLearnedPositionalEmbedding, OPTAttention,
     KwargsForCausalLM, FlashAttentionKwargs
)
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import logging, is_torch_flex_attn_available
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

# Import C++ extensions
from sparse_transformers import (
    sparse_mlp_forward_opt,
    WeightCacheOpt,
    approx_topk_threshold
)

from src.models.opt.configuration_opt_skip import OPTSkipConnectionConfig
from src.modeling_skip import (
    SkipMLP, SkipDecoderLayer,
    build_skip_connection_model, build_skip_connection_model_for_causal_lm
)

logger = logging.get_logger(__name__)


class OPTMLP(nn.Module):    # double check config stuff later
    def __init__(self, config):
        self.up_proj = nn.Linear(config.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.down_proj = nn.Linear(config.ffn_dim, config.embed_dim, bias=config.enable_bias)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.up_proj(x))
        return self.down_proj(x)


class OPTSkipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, sparsity: float, bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.sparsity = sparsity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Initialize mask but defer WeightCache creation until post_init
        self.init_mask = torch.ones(intermediate_size, dtype=torch.bool)
        self.init_mask[int(intermediate_size * sparsity):] = 0
        
        self.weight_cache = None

        # Register buffers - start with reasonable size and ensure they can be resized
        self.register_buffer('down_proj_buffer', torch.zeros(1, hidden_size, requires_grad=False))
        self.register_buffer('up_proj_buffer', torch.zeros(1, int(intermediate_size * sparsity), requires_grad=False))

    def initialize_weight_cache(self):
        """Tie weights after weights are loaded (called from post_init)."""
        if self.weight_cache is None:
            # Create and initialize weight cache
            self.weight_cache = WeightCacheOpt(   
                self.init_mask,
                self.hidden_size,
                self.up_proj.weight, 
                self.down_proj.weight
            )

    def to(self, *args, **kwargs):
        # Move buffers to same device as model when .to() is called
        result = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device')
        if device:
            self.down_proj_buffer = self.down_proj_buffer.to(device)
            self.up_proj_buffer = self.up_proj_buffer.to(device)
            if hasattr(self, 'init_mask'):
                self.init_mask = self.init_mask.to(device)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sparse_mlp_forward_opt(
            x.detach(), 
            self.weight_cache.get_active_up_weight(),
            self.weight_cache.get_active_down_weight(),
            self.down_proj_buffer,
            self.up_proj_buffer,
            "relu"
        )
        return out
    

class OPTSkipDecoderLayer(SkipDecoderLayer):
    def _init_components(self, config: OPTSkipConnectionConfig, layer_idx: int):
        self.self_attn = OPTAttention(config=config, layer_idx=layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout

        # Needed in order to load weights, will be deleted after
        self.fc1 = nn.Linear(self.hidden_size, config.intermediate_size, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.intermediate_size, self.hidden_size, bias=config.enable_bias) 

    def _fix_unloaded_weights(self):    # check if this works - not sure if this will be called by the loop in from pretrained
        self.mlp.up_proj.load_state_dict({'weight': self.fc1.weight, 'bias': self.fc1.bias}, assign=True)
        self.mlp.down_proj.load_state_dict({'weight': self.fc2.weight, 'bias': self.fc2.bias}, assign=True)
        del self.fc1
        del self.fc2

    def _set_mlp_train(self, config: OPTSkipConnectionConfig):
        self.mlp = OPTMLP(config)

    def _set_mlp_inference(self, config: OPTSkipConnectionConfig):
        self.mlp = OPTSkipMLP(
            config.hidden_size,
            config.intermediate_size,
            config.sparsity,
            config.enable_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],    
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *OPTional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *OPTional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *OPTional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *OPTional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *OPTional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *OPTional*):
                Indices depicting the position of the input sequence tokens in the sequence..
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if not self.training:  # Use PyTorch's built-in training flag
            self._compute_binary_mask(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_ids=position_ids,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

class OPTSkipPreTrainedModel(PreTrainedModel):
    config_class = OPTSkipConnectionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTSkipDecoderLayer"]
    _supports_attention_backend = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


OPTSkipConnectionModelBase: type[OPTSkipPreTrainedModel] = build_skip_connection_model(OPTSkipPreTrainedModel)

class OPTSkipConnectionModel(OPTSkipConnectionModelBase):
    def _init_components(self, config: OPTSkipConnectionConfig):
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OPTSkipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None
        #self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.rotary_emb = MistralRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`. for padding use -1.

                [What are position IDs?](../glossary#position-ids)
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
                the complete sequence length.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `DynamicCache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        if position_ids is None:
            # position_ids = cache_position.unsqueeze(0)
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_seen_tokens` is > 0
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                # Collect predictor loss if available

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    

class OPTSkipDecoderWrapper(OPTSkipPreTrainedModel):
    def __init__(self, config: OPTSkipConnectionConfig):
        super().__init__(config)
        self.decoder = OPTSkipConnectionModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder
    
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        return self.decoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            head_mask=head_mask,
            return_dict=return_dict,
            **kwargs,
        )



class OPTSkipConnectionForCausalLM(OPTSkipPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [
        "model.decoder.layers.*.mlp.up_proj",
        "model.decoder.layers.*.mlp.down_proj",
        "model.decoder.layers.*.mlp.combined_proj_buffer",
        "model.decoder.layers.*.mlp.down_proj_buffer",
        "model.decoder.layers.*.mlp.init_mask",
        "model.decoder.layers.*.mlp.weight_cache",
        "model.decoder.layers.*.mlp_lora_proj.down.weight",
        "model.decoder.layers.*.mlp_lora_proj.intermediate",
        "model.decoder.layers.*.mlp_lora_proj.output", 
        "model.decoder.layers.*.mlp_lora_proj.up.weight",
        "model.decoder.layers.*.mlp_mask",
        "model.decoder.layers.*.standard_mlp.gate_proj.weight",
        "model.decoder.layers.*.standard_mlp.up_proj.weight",
        "model.decoder.layers.*.standard_mlp.down_proj.weight"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTSkipDecoderWrapper(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        out = super(OPTSkipConnectionForCausalLM, cls).from_pretrained(*args, **kwargs)
        for module in out.modules():
            if any(hasattr(p, 'is_meta') and p.is_meta for p in module.parameters()) and \
                    hasattr(module, '_fix_unloaded_weights'):
                module = module._fix_unloaded_weights()
        return out

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder
    
    def get_predictor_parameters(self):
        """Get parameters of all predictor networks for optimization."""
        return self.model.decoder.get_predictor_parameters()

    def freeze_non_predictor_parameters(self):
        """Freeze all parameters except predictor networks."""
        # Freeze LM head
        for param in self.lm_head.parameters():
            param.requires_grad = False
        
        # Freeze model parameters except predictors
        self.model.decoder.freeze_non_predictor_parameters()

    def reset_cache(self):
        """Reset cache of all layers."""
        for layer in self.model.decoder.layers:
            layer.mlp.weight_cache = None
            layer.mlp.initialize_weight_cache()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            head_mask=head_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


#OPTSkipConnectionForCausalLMBase: type[OPTSkipPreTrainedModel] = \
#    build_skip_connection_model_for_causal_lm(OPTSkipPreTrainedModel, OPTSkipConnectionModel)

#class OPTSkipConnectionForCausalLM(OPTSkipConnectionForCausalLMBase):
#    _keys_to_ignore_on_load_missing = [
#        "model.layers.*.mlp.combined_proj_buffer",
#        "model.layers.*.mlp.down_proj_buffer",
#        "model.layers.*.mlp.init_mask",
#        "model.layers.*.mlp.weight_cache",
#        "model.layers.*.mlp_lora_proj.down.weight",
#        "model.layers.*.mlp_lora_proj.intermediate",
#        "model.layers.*.mlp_lora_proj.output", 
#        "model.layers.*.mlp_lora_proj.up.weight",
#        "model.layers.*.mlp_mask",
#        "model.layers.*.mlp.gate_proj.weight",
#        "model.layers.*.mlp.up_proj.weight",
#        "model.layers.*.standard_mlp.gate_proj.weight",
#        "model.layers.*.standard_mlp.up_proj.weight",
#        "model.layers.*.standard_mlp.down_proj.weight"
#    ]
