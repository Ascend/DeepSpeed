import copy
import datetime
import json
import pathlib
from typing import Tuple, List

import torch
import torch_npu
import transformers
import deepspeed_npu
import deepspeed
import fire
import t5_utils
import t5_patch

from functools import partial
from collections import namedtuple
from torch import Tensor, IntTensor
from torch.utils.data import DataLoader
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils.import_utils import is_torch_fx_proxy
from transformers.models.t5.modeling_t5 import T5DenseReluDense, T5DenseGatedGeluDense, T5Attention
from transformers.models.t5.modeling_t5 import T5Block, T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5LayerNorm

InputData = namedtuple('InputData', ['input_ids', 'labels', 'attention_mask', 'encoder_hidden_states',
                                     'encoder_attention_mask', 'inputs_embeds', 'head_mask',
                                     'cross_attn_head_mask', 'past_key_values', 'use_cache',
                                     'output_attentions', 'output_hidden_states', 'return_dict',
                                     'hidden_states', 'encoder_extended_attention_mask', 'position_bias',
                                     'encoder_decoder_position_bias', 'decoder_input_ids',
                                     'decoder_inputs_embeds', 'lm_logits', 'loss', 'extended_attention_mask',
                                     'decoder_attention_mask', 'decoder_head_mask', 'input_shape'])


class T5FirstLayerPipeline(torch.nn.Module):
    def forward(self, inputs):
        device = inputs[0].device
        # use IntTensor because need is_floating_point() is false
        input_data = InputData(inputs[0], inputs[1], inputs[2], IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               IntTensor().to(device), IntTensor().to(device), IntTensor().to(device),
                               inputs[3], IntTensor().to(device), IntTensor().to(device))
        return input_data


class T5BeforeEmbeddingPipeline(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)
        device = inputs.input_ids.device
        inputs = inputs._replace(
            use_cache=(
                inputs.use_cache if inputs.use_cache.numel() != 0 else IntTensor([self.config.use_cache]).to(device)),
            output_attentions=(inputs.output_attentions if inputs.output_attentions.numel() != 0 else IntTensor(
                [self.config.output_attentions]).to(device)),
            output_hidden_states=(
                inputs.output_hidden_states if inputs.output_hidden_states.numel() != 0 else IntTensor(
                    [self.config.output_hidden_states]).to(device)),
            return_dict=(
                inputs.return_dict if inputs.return_dict.numel() != 0 else IntTensor([self.config.use_return_dict]).to(
                    device))
        )
        if inputs.input_ids.numel() != 0 and inputs.inputs_embeds.numel() != 0:
            err_msg_prefix = 'decoder_' if self.config.is_decoder else ''
            raise ValueError(
                f'You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time')
        elif inputs.input_ids.numel() != 0:
            input_shape = inputs.input_ids.size()
            inputs = inputs._replace(input_ids=inputs.input_ids.view(-1, input_shape[-1]))
        elif inputs.inputs_embeds.numel() != 0:
            input_shape = inputs.inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = 'decoder_' if self.config.is_decoder else ''
            raise ValueError(f'You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds')

        # move embedding to next layer

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = inputs.past_key_values[0][0].shape[
                              2] + seq_length if inputs.past_key_values.numel() != 0 else seq_length

        if inputs.use_cache[0].item() == 1:  # inputs.use_cache is True:
            assert self.config.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if inputs.attention_mask.numel() == 0:
            inputs = inputs._replace(
                attention_mask=torch.ones(batch_size, mask_seq_length).to(dtype=torch.int32, device=device))
        if self.config.is_decoder and inputs.encoder_attention_mask.numel() == 0 and inputs.encoder_hidden_states.numel() != 0:
            encoder_seq_length = inputs.encoder_hidden_states.shape[1]
            inputs = inputs._replace(
                encoder_attention_mask=torch.ones(batch_size, encoder_seq_length, device=device, dtype=torch.int))

        # initialize past_key_values with `None` if past does not exist
        # if inputs.past_key_values.numel() == 0:
        #     inputs = inputs._replace(past_key_values=([None] * self.config.num_layers))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        inputs = inputs._replace(
            extended_attention_mask=self.get_extended_attention_mask(inputs.attention_mask, input_shape,
                                                                     inputs.inputs_embeds.device,
                                                                     inputs.attention_mask.dtype))

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and inputs.encoder_hidden_states.numel() != 0:
            encoder_batch_size, encoder_sequence_length, _ = inputs.encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if inputs.encoder_attention_mask.numel() == 0:
                inputs = inputs._replace(encoder_attention_mask=torch.ones(encoder_hidden_shape, dtype=torch.int32,
                                                                           device=inputs.inputs_embeds.device))
            inputs = inputs._replace(
                encoder_extended_attention_mask=self.invert_attention_mask(inputs.encoder_attention_mask,
                                                                           inputs.encoder_attention_mask.dtype))
        else:
            inputs = inputs._replace(encoder_extended_attention_mask=IntTensor().to(inputs.input_ids.device))

        # Prepare head mask if needed
        # dont need, just were outputs
        # inputs.present_key_value_states = torch.Tensor() if inputs.use_cache else None
        # inputs.all_hidden_states = torch.Tensor() if inputs.output_hidden_states else None
        # inputs.all_attentions = torch.Tensor() if inputs.output_attentions else None
        # inputs.all_cross_attentions = torch.Tensor() if inputs.output_attentions and self.config.is_decoder else None
        inputs = inputs._replace(position_bias=IntTensor().to(inputs.input_ids.device))
        inputs = inputs._replace(encoder_decoder_position_bias=IntTensor().to(inputs.input_ids.device))

        return inputs

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device, dtype) -> Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(input_shape, attention_mask,
                                                                                          device)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f'Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})')

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = (1 - extended_attention_mask) * -10000
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor, dtype) -> Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility

        if dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        elif dtype == torch.int32:
            encoder_extended_attention_mask = ((1 - encoder_extended_attention_mask) * -1e4).int()
        else:
            raise ValueError(
                f'{dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`')

        return encoder_extended_attention_mask


class T5EmbeddingPipeline(torch.nn.Embedding):
    def __init__(self, config):
        super(T5EmbeddingPipeline, self).__init__(config.vocab_size, config.d_model)
        self.config = config

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)
        if inputs.inputs_embeds.numel() == 0:
            inputs = inputs._replace(hidden_states=super().forward(inputs.input_ids))
        return inputs


class T5DropoutPipeline(torch.nn.Dropout):
    def __init__(self, dropout_rate):
        super(T5DropoutPipeline, self).__init__(dropout_rate)

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)
        if inputs.inputs_embeds.numel() != 0:
            inputs = inputs._replace(hidden_states=super().forward(inputs.inputs_embeds))
            inputs = inputs._replace(inputs_embeds=IntTensor().to(inputs.inputs_embeds.device))
        else:
            inputs = inputs._replace(hidden_states=super().forward(inputs.hidden_states))
        return inputs


class T5BlockPipeline(T5Block):
    def __init__(self, config, idx, layer_head_mask, cross_attn_layer_head_mask, past_key_value):
        super(T5BlockPipeline, self).__init__(config, has_relative_attention_bias=bool(idx == 0))
        self.idx = idx
        self.config = config
        self.layer_head_mask = layer_head_mask
        self.cross_attn_layer_head_mask = cross_attn_layer_head_mask
        self.past_key_value = past_key_value
        if self.is_decoder:
            self.layer[1].EncDecAttention.gradient_checkpointing = True

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)
        # model parallel, we dont need
        # dont need, just is output
        # if inputs.output_hidden_states[0].item():
        #     inputs.all_hidden_states = torch.cat((inputs.all_hidden_states, inputs.hidden_states))
        # gradient checkpoint, we dont need
        layer_outputs = super().forward(
            inputs.hidden_states,
            attention_mask=inputs.extended_attention_mask,
            position_bias=None if inputs.position_bias.numel() == 0 else inputs.position_bias,
            encoder_hidden_states=None if inputs.encoder_hidden_states.numel() == 0 else inputs.encoder_hidden_states,
            encoder_attention_mask=None if inputs.encoder_extended_attention_mask.numel() == 0 else inputs.encoder_extended_attention_mask,
            encoder_decoder_position_bias=None if inputs.encoder_decoder_position_bias.numel() == 0 else inputs.encoder_decoder_position_bias,
            layer_head_mask=self.layer_head_mask,
            cross_attn_layer_head_mask=self.cross_attn_layer_head_mask,
            past_key_value=self.past_key_value,
            use_cache=True if inputs.use_cache[0].item() else False,
            output_attentions=True if inputs.output_attentions[0].item() else False,
        )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if inputs.use_cache[0].item() == 0:  # inputs.use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
        hidden_states, present_key_value_state = layer_outputs[:2]
        inputs = inputs._replace(hidden_states=hidden_states)

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)

        inputs = inputs._replace(position_bias=layer_outputs[2].contiguous())
        # inputs = inputs._replace(position_bias=IntTensor().to(inputs.hidden_states.device))
        if self.is_decoder and inputs.encoder_hidden_states.numel() != 0:
            inputs = inputs._replace(
                encoder_decoder_position_bias=layer_outputs[4 if inputs.output_attentions[0].item() else 3])
            # inputs = inputs._replace(encoder_decoder_position_bias=IntTensor().to(inputs.hidden_states.device))

        # append next layer key value states
        # dont need, just is output
        # if inputs.use_cache[0].item():
        #     inputs.present_key_value_states = inputs.present_key_value_states + (present_key_value_state,)
        # if inputs.output_attentions:
        #     inputs.all_attentions = inputs.all_attentions + (layer_outputs[3],)
        #     if self.is_decoder:
        #         inputs.all_cross_attentions = inputs.all_cross_attentions + (layer_outputs[5],)

        # model parallel, we dont need
        if self.idx + 1 == self.config.num_layers:
            inputs = inputs._replace(
                encoder_hidden_states=IntTensor().to(inputs.use_cache.device),
                position_bias=IntTensor().to(inputs.hidden_states.device),
                encoder_decoder_position_bias=IntTensor().to(inputs.hidden_states.device)
            )

        return inputs


class T5LayerNormPipeline(T5LayerNorm):
    def __init__(self, d_model, eps):
        super(T5LayerNormPipeline, self).__init__(d_model, eps=eps)

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)

        inputs = inputs._replace(hidden_states=super().forward(inputs.hidden_states))
        return inputs


class T5LastEncoderLayerPipeline(torch.nn.Module):
    def __init__(self, config):
        super(T5LastEncoderLayerPipeline, self).__init__()
        self.config = config

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
                    decoder_start_token_id is not None), 'self.model.config.decoder_start_token_id has to be defined.' \
                                                         ' In T5 it is usually set to the pad_token_id. ' \
                                                         'See T5 docs for more information'

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.full(input_ids.shape, decoder_start_token_id,
                                               dtype=input_ids.dtype, device=input_ids.device)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()

        assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'

        # don't need, because collate_function does it
        # replace possible -100 values in labels by `pad_token_id`
        # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        # assert torch.all(shifted_input_ids >= 0).item(), 'Verify that `shifted_input_ids` has only positive values'

        return shifted_input_ids

    def forward(self, inputs):
        if not isinstance(inputs, InputData):
            inputs = InputData(*inputs)
        # dont need, just is output
        # if inputs.output_hidden_states[0].item():
        #     inputs.all_hidden_states = inputs.all_hidden_states + (inputs.hidden_states,)
        if inputs.labels.numel() != 0 and inputs.decoder_input_ids.numel() == 0 and inputs.decoder_inputs_embeds.numel() == 0:
            # get decoder inputs from shifting lm labels to the right
            inputs = inputs._replace(decoder_input_ids=self._shift_right(inputs.labels))

        # model parallel, we dont need

        inputs = inputs._replace(
            input_ids=inputs.decoder_input_ids,
            attention_mask=inputs.decoder_attention_mask,
            inputs_embeds=inputs.decoder_inputs_embeds,
            encoder_hidden_states=inputs.hidden_states,
            encoder_attention_mask=inputs.attention_mask,
            head_mask=inputs.decoder_head_mask,
            use_cache=IntTensor().to(inputs.use_cache.device),  # use to config.use_cache
            return_dict=inputs.return_dict,
            hidden_states=IntTensor().to(inputs.use_cache.device)
        )
        return inputs


def t5_lmhead_forward(layer, inputs):
    if not isinstance(inputs, InputData):
        inputs = InputData(*inputs)
    if layer.config.tie_word_embeddings:
        inputs = inputs._replace(hidden_states=inputs.hidden_states * (layer.config.d_model ** -0.5))

    bias = getattr(layer, "bias", None)
    if bias is not None:
        bias = bias.copy()
        bias.data = torch.nn.functional.pad(layer.bias.data, (0, layer.weight.shape[0] - layer.bias.shape[0],),
                                            "constant", 0, )

    output = torch.nn.functional.linear(inputs.hidden_states, layer.weight, bias)
    inputs = inputs._replace(hidden_states=output)
    return inputs


def loss_fn_pipe(inputs, labels):
    if not isinstance(inputs, InputData):
        inputs = InputData(*inputs)
    if inputs.labels.numel() != 0:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)  # is T5 tokenizer pad_token_id
        loss = loss_fct(inputs.hidden_states.view(-1, inputs.hidden_states.size(-1)), inputs.labels.view(-1))
        return loss
    return None


class T5Pipeline(PipelineModule):
    def __init__(self, config: T5Config, num_stages=2):
        self.config = config
        self.model_dim = config.d_model
        self.specs = []

        self.specs.append(LayerSpec(T5FirstLayerPipeline))

        # encoder
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.specs.append(LayerSpec(T5BeforeEmbeddingPipeline, encoder_config))
        self.specs.append(TiedLayerSpec('embedding', T5EmbeddingPipeline, encoder_config))
        self.specs.append(LayerSpec(T5DropoutPipeline, config.dropout_rate))
        for i in range(config.num_layers):
            # layer_head_mask and cross_attn_layer_head_mask, if not None set here
            self.specs.append(LayerSpec(T5BlockPipeline, encoder_config, i, None, None, None))
        self.specs.append(LayerSpec(T5LayerNormPipeline, config.d_model, eps=config.layer_norm_epsilon))
        self.specs.append(LayerSpec(T5DropoutPipeline, config.dropout_rate))

        self.specs.append(LayerSpec(T5LastEncoderLayerPipeline, encoder_config))

        # decoder
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        self.specs.append(LayerSpec(T5BeforeEmbeddingPipeline, decoder_config))
        self.specs.append(TiedLayerSpec('embedding', T5EmbeddingPipeline, decoder_config))
        self.specs.append(LayerSpec(T5DropoutPipeline, config.dropout_rate))
        for i in range(config.num_layers):
            # layer_head_mask and cross_attn_layer_head_mask, if is not None set here
            self.specs.append(LayerSpec(T5BlockPipeline, decoder_config, i, None, None, None))
        self.specs.append(LayerSpec(T5LayerNormPipeline, config.d_model, eps=config.layer_norm_epsilon))
        self.specs.append(LayerSpec(T5DropoutPipeline, config.dropout_rate))
        self.specs.append(TiedLayerSpec('embedding', T5EmbeddingPipeline, config, forward_fn=t5_lmhead_forward))

        super().__init__(layers=self.specs, num_stages=num_stages, loss_fn=loss_fn_pipe, partition_method='greedy')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""

        torch.manual_seed(123)
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5EmbeddingPipeline)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            w = torch.normal(mean=0.0, std=factor * 1.0, size=module.weight.shape)
            module.weight.data = torch.nn.Parameter(w)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            # module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wi.weight.shape)
            module.wi.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            # module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5), size=module.wo.weight.shape)
            module.wo.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            # module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wi_0.weight.shape)
            module.wi_0.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            # module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wi_1.weight.shape)
            module.wi_1.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            # module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5), size=module.wo.weight.shape)
            module.wo.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            # module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            # module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            # module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            # module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            module.q.weight.data = torch.nn.Parameter(
                torch.normal(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5),
                             size=module.q.weight.shape))
            module.k.weight.data = torch.nn.Parameter(
                torch.normal(mean=0.0, std=factor * (d_model ** -0.5), size=module.k.weight.shape))
            module.v.weight.data = torch.nn.Parameter(
                torch.normal(mean=0.0, std=factor * (d_model ** -0.5), size=module.v.weight.shape))
            module.o.weight.data = torch.nn.Parameter(
                torch.normal(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5),
                             size=module.o.weight.shape))

            if module.has_relative_attention_bias:
                # module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
                w = torch.normal(mean=0.0, std=factor * ((d_model) ** -0.5),
                                 size=module.relative_attention_bias.weight.shape)
                module.relative_attention_bias.weight.data = torch.nn.Parameter(w)


######################################################################
########### Huggingface Transformers Related Functions ###############
######################################################################

def collate_function(batch: List[Tuple[List[int], List[int]]], pad_token_id: int, max_length: int):
    padded_token_ids = [token_ids + [pad_token_id for _ in range(0, max_length - len(token_ids))]
                        for token_ids, _ in batch]
    padded_labels = [labels + [pad_token_id for _ in range(0, max_length - len(labels))] for _, labels in batch]

    src_tokens = torch.IntTensor(padded_token_ids)
    tgt_tokens = torch.IntTensor(padded_labels)

    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    decoder_attention_mask = tgt_tokens.ne(pad_token_id).type_as(tgt_tokens)
    return ((src_tokens, tgt_tokens, attention_mask, decoder_attention_mask), tgt_tokens)


def create_collate_fn(tokenizer, max_length: int):
    collate_fn_partial = partial(collate_function, pad_token_id=tokenizer.pad_token_id, max_length=max_length)
    return collate_fn_partial


def create_pipeline_model(num_stages, decoder_start_token_id, dropout_rate, n_positions, num_layers, num_heads, ff_dim, d_model):
    config = T5Config(n_positions=n_positions, output_past=True)
    config.dropout_rate = dropout_rate
    config.decoder_start_token_id = decoder_start_token_id
    config.moe_enabled = False
    if num_layers is not None:
        config.num_layers = num_layers
    if num_heads is not None:
        config.num_heads = num_heads
    if ff_dim is not None:
        config.d_ff = ff_dim
    if d_model is not None:
        config.d_model = d_model
    return T5Pipeline(config, num_stages=num_stages)


class BatchPipelineTruncate:
    def __init__(self, model):
        self.model = model

    def get_batch_pipe(self, data):
        if hasattr(self.model, 'curriculum_scheduler'):
            inputs, labels = data
            seq_len = self.model.curriculum_scheduler.get_current_difficulty()
            if len(inputs[0][0]) > seq_len:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i][:, :seq_len].contiguous()
                return (inputs, inputs[1])
        return data


def train(
        checkpoint_dir: str = None,
        load_checkpoint_dir: str = None,

        # Dataset Params
        dataset_dir: str = None,
        mask_prob: float = 0.15,
        random_replace_span: int = 3,
        max_seq_length: int = 1024,
        tokenizer_name_or_dir: str = 't5-small',

        # Model Params
        num_layers: int = None,
        num_heads: int = None,
        ff_dim: int = None,
        n_positions: int = 512,
        d_model: int = 512,
        dropout: float = 0.1,

        # Training Params
        batch_size: int = 8,
        num_iterations: int = 1000000,
        checkpoint_every: int = -1,

        # DeepSpeed Params
        num_stage: int = 1,
        fp16: bool = False,
        initial_scale_power: int = 32,
        zero_stage: int = 0,
        local_rank: int = -1
):
    deepspeed.init_distributed(dist_backend='hccl')

    if checkpoint_dir is None and load_checkpoint_dir is None:
        t5_utils.log_dist('Need to specify one of checkpoint_dir or load_checkpoint_dir', ranks=[0],
                          level=t5_utils.logging.ERROR)
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        t5_utils.log_dist('Cannot specify both checkpoint_dir and load_checkpoint_dir', ranks=[0],
                          level=t5_utils.logging.ERROR)
        return
    if checkpoint_dir:
        t5_utils.log_dist('Creating Experiment Directory', ranks=[0], level=t5_utils.logging.INFO)
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            # Dataset Params
            'mask_prob': mask_prob,
            'random_replace_span': random_replace_span,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer_name_or_dir,

            # Model Params
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'n_positions': n_positions,
            'd_model': d_model,
            'dropout': dropout,

            # Training Params
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'checkpoint_every': checkpoint_every,

            # DeepSpeed Params
            'num_stage': num_stage,
            'fp16': fp16,
            'initial_scale_power': initial_scale_power,
            'zero_stage': zero_stage,

        }
        exp_dir = t5_utils.create_experiment_dir(checkpoint_dir, all_arguments)
        t5_utils.log_dist(f'Experiment Directory created at {exp_dir}', ranks=[0], level=t5_utils.logging.INFO)
    else:
        t5_utils.log_dist('Loading from Experiment Directory', ranks=[0], level=t5_utils.logging.INFO)
        load_checkpoint_dir = pathlib.Path(load_checkpoint_dir)
        assert load_checkpoint_dir.exists()
        with (load_checkpoint_dir / 'hparams.json').open('r') as handle:
            hparams = json.load(handle)
        # Dataset Params
        mask_prob = hparams.get('mask_prob', mask_prob)
        random_replace_span = hparams.get('random_replace_span', random_replace_span)
        max_seq_length = hparams.get('max_seq_length', max_seq_length)
        tokenizer_name_or_dir = hparams.get('tokenizer', tokenizer_name_or_dir)

        # Model Params
        num_layers = hparams.get('num_layers', num_layers)
        num_heads = hparams.get('num_heads', num_heads)
        ff_dim = hparams.get('ff_dim', ff_dim)
        n_positions = hparams.get('n_positions', n_positions)
        d_model = hparams.get('d_model', d_model)
        dropout = hparams.get('dropout', dropout)

        # Training Params
        batch_size = hparams.get('batch_size', batch_size)
        _num_iterations = hparams.get('num_iterations', num_iterations)
        num_iterations = max(num_iterations, _num_iterations)
        checkpoint_every = hparams.get('checkpoint_every', checkpoint_every)

        # DeepSpeed Params
        num_stage = hparams.get('num_stage', num_stage)
        fp16 = hparams.get('fp16', fp16)
        initial_scale_power = hparams.get('initial_scale_power', initial_scale_power)
        zero_stage = hparams.get('zero_stage', zero_stage)

        exp_dir = load_checkpoint_dir

    ds_config = {
        'train_batch_size': batch_size,
        'optimizer': {
            'type': 'Adam',
            'params': {
                'lr': 1e-4,
                'torch_adam': True
            }
        },
        # in pipeline, the batch size need be same, so we need drop last batch in dataloader
        'dataloader_drop_last': True,
        'curriculum_learning': {
            'enabled': False,
            'curriculum_type': 'seqlen',
            'min_difficulty': 8,
            'max_difficulty': 1024,
            'schedule_type': 'fixed_linear',
            'schedule_config': {
                'total_curriculum_step': 10000,
                'difficulty_step': 8
            }
        },
        'fp16': {
            'enabled': fp16,
            'initial_scale_power': initial_scale_power
        },
        'zero_optimization': {
            'stage': zero_stage,
            'reduce_bucket_size': 5e8
        },
        'checkpoint': {
            'tag_validation': 'Ignore'
        }
    }

    t5_utils.log_dist('Creating Tokenizer', ranks=[0], level=t5_utils.logging.INFO)
    tokenizer = t5_utils.create_tokenizer(tokenizer_name_or_dir)

    t5_utils.log_dist('Creating Datasets', ranks=[0], level=t5_utils.logging.INFO)
    dataset = t5_utils.create_dataset(tokenizer=tokenizer, dataset_dir=dataset_dir, mask_prob=mask_prob,
                                      random_replace_span=random_replace_span, max_seq_length=max_seq_length)
    collate_fn = create_collate_fn(tokenizer, max_length=max_seq_length)
    t5_utils.log_dist('Dataset Creation Done with length: {}'.format(len(dataset)), ranks=[0],
                      level=t5_utils.logging.INFO)

    t5_utils.log_dist('Creating Model', ranks=[0], level=t5_utils.logging.INFO)
    t5_patch.t5_performance_optimize()
    model = create_pipeline_model(num_stage, tokenizer.added_tokens_encoder['<decoder>'],
                                  dropout, n_positions, num_layers, num_heads, ff_dim, d_model)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config,
                                          training_data=dataset, collate_fn=collate_fn)
    # Curriculum Learning batch function
    # batch_pipe = BatchPipelineTruncate(model)
    # model.set_batch_fn(batch_pipe.get_batch_pipe)

    t5_utils.log_dist('DeepSpeed engine created', ranks=[0], level=t5_utils.logging.INFO)
    t5_utils.log_dist(f'Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}', ranks=[0],
                      level=t5_utils.logging.INFO)

    # Load Model checkpoint
    start_step = 0
    if load_checkpoint_dir is not None:
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        checkpoint_step = client_state['checkpoint_step']
        start_step = checkpoint_step + 1

    # The Training Loop
    model.train()
    last_losses = []
    for step in range(start_step, num_iterations):
        start_time = datetime.datetime.now()
        loss = model.train_batch()
        t5_utils.log_dist(
            'Step: {}\tDuration: {}\tLoss: {}'.format(step, datetime.datetime.now() - start_time, loss.item()),
            ranks=[0], level=t5_utils.logging.INFO)

        if step != 0 and checkpoint_every > 0 and step % checkpoint_every == 0:
            model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step}, tag='step_{}'.format(step))
            t5_utils.log_dist("Saved model to {0}".format(exp_dir), ranks=[0], level=t5_utils.logging.INFO)

        if step > num_iterations - 1000:
            last_losses.append(loss.item())
        if step >= num_iterations:
            break

    # Save the last checkpoint if not saved yet
    if checkpoint_every > 0 and step % checkpoint_every != 0:
        model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step}, tag='step_{}'.format(step))
        t5_utils.log_dist("Saved model to {0}".format(exp_dir), ranks=[0], level=t5_utils.logging.INFO)

    t5_utils.log_dist('finished training, last {} losses average: {}'
                      .format(len(last_losses), sum(last_losses) / len(last_losses)),
                      ranks=[0], level=t5_utils.logging.INFO)

    return exp_dir


if __name__ == '__main__':
    option = {}
    option["ACL_OP_COMPILER_CACHE_MODE"] = "enable"  # cache功能启用
    option["ACL_OP_COMPILER_CACHE_DIR"] = "./cache"  # cache所在文件夹

    print("option:", option)
    torch.npu.set_option(option)
    transformers.set_seed(123)
    torch.npu.manual_seed_all(123)
    fire.Fire(train)
