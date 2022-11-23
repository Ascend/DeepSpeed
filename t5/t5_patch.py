import torch
import torch_npu
import math
from transformers.models.t5.modeling_t5 import T5Attention, T5Block, T5PreTrainedModel, T5LayerNorm, logger
from transformers.utils.import_utils import is_torch_fx_proxy


def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.int) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
    ).to(torch.int)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets


def compute_bias(self, query_length, key_length):
    """Compute binned relative position bias"""
    context_position = torch.arange(
        query_length, dtype=torch.int, device=self.relative_attention_bias.weight.device
    )[:, None]
    memory_position = torch.arange(
        key_length, dtype=torch.int, device=self.relative_attention_bias.weight.device
    )[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values


def t5_block_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
):
    if past_key_value is not None:
        if not self.is_decoder:
            logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
        expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

        if len(past_key_value) != expected_num_past_key_values:
            raise ValueError(
                f"There should be {expected_num_past_key_values} past states. "
                f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                f"Got {len(past_key_value)} past key / value states"
            )

        self_attn_past_key_value = past_key_value[:2]
        cross_attn_past_key_value = past_key_value[2:]
    else:
        self_attn_past_key_value, cross_attn_past_key_value = None, None

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=self_attn_past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if present_key_value_state is not None:
            query_length = present_key_value_state[0].shape[2]
        else:
            query_length = None

        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = cross_attention_outputs[0]

        # Combine self attn and cross attn key value states
        if present_key_value_state is not None:
            present_key_value_state = present_key_value_state + cross_attention_outputs[1]

        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (present_key_value_state,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def _shift_right(self, input_ids):
    decoder_start_token_id = self.config.decoder_start_token_id
    pad_token_id = self.config.pad_token_id

    assert (
            decoder_start_token_id is not None
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. " \
       "See T5 docs for more information"

    # shift inputs to the right
    if is_torch_fx_proxy(input_ids):
        # Item assignment is not supported natively for proxies.
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
    else:
        # shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        # shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        # shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids = torch.full(input_ids.shape, decoder_start_token_id,
                                       dtype=input_ids.dtype, device=input_ids.device)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

    # don't need, because t5_model.py:collate_function does it
    # replace possible -100 values in labels by `pad_token_id`
    # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    # assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


def t5_performance_optimize():
    T5Block.forward = t5_block_forward
    T5Attention._relative_position_bucket = _relative_position_bucket
    T5Attention.compute_bias = compute_bias

    T5PreTrainedModel._shift_right = _shift_right
