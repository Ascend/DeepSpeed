import datetime
import json
import math
import pathlib
import warnings
import fire
import deepspeed_npu
import deepspeed
import torch
import torch_npu
import transformers
import t5_utils

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, T5Model
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import T5DenseReluDense, __HEAD_MASK_WARNING_MSG, T5LayerNorm, T5EncoderModel, \
    T5DenseGatedGeluDense, T5Attention, T5PreTrainedModel
from torch import nn
from functools import partial
from typing import Tuple, List, Optional, Union


def collate_function(batch: List[Tuple[List[int], List[int]]], pad_token_id: int):
    max_length = 1024
    padded_token_ids = [token_ids + [pad_token_id for _ in range(0, max_length - len(token_ids))]
                        for token_ids, _ in batch]
    padded_labels = [labels + [pad_token_id for _ in range(0, max_length - len(labels))] for _, labels in batch]

    src_tokens = torch.LongTensor(padded_token_ids)
    tgt_tokens = torch.LongTensor(padded_labels)

    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    decoder_attention_mask = tgt_tokens.ne(pad_token_id).type_as(tgt_tokens)
    return (src_tokens, tgt_tokens, attention_mask, decoder_attention_mask)


def create_collate_fn(tokenizer):
    collate_fn_partial = partial(collate_function, pad_token_id=tokenizer.pad_token_id)
    return collate_fn_partial


class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, masking_function):
        self.dataset = dataset
        self.masking_function = masking_function

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        tokens, labels = self.masking_function(self.dataset[idx]['text'])
        return (tokens, labels)


def T5DenseReluDenseInit(self, config: T5Config):
    super(T5DenseReluDense, self).__init__()
    self.moe_enabled = config.moe_enabled if hasattr(config, 'moe_enabled') else False
    if self.moe_enabled:
        self.moe_num_experts = config.moe_num_experts
        self.moe_ep_size = config.moe_ep_size
        self.wi = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wi = deepspeed.moe.layer.MoE(hidden_size=config.d_model, expert=self.wi, num_experts=self.moe_num_experts,
                                          ep_size=self.moe_ep_size, use_rts=True)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
    else:
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.dropout = nn.Dropout(config.dropout_rate)


def T5DenseReluDenseForward(self, hidden_states):
    hidden_states = self.wi(hidden_states)
    if self.moe_enabled:
        hidden_states = self.wi_1(hidden_states[0])  # MoE return a tuple, [0] is the output
    hidden_states = nn.functional.relu(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


def InitWeights(self, module):
    """Initialize the weights"""
    torch.manual_seed(123)
    factor = self.config.initializer_factor  # Used for testing weights initialization
    if isinstance(module, T5LayerNorm):
        module.weight.data.fill_(factor * 1.0)
    elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
        # Mesh TensorFlow embeddings initialization
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
        w = torch.normal(mean=0.0, std=factor * 1.0, size=module.shared.weight.shape)
        module.shared.weight.data = torch.nn.Parameter(w)
    elif isinstance(module, T5DenseReluDense):
        # Mesh TensorFlow FF initialization
        # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
        # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
        if isinstance(module.wi, deepspeed.moe.layer.MoE):
            for layer in module.wi.deepspeed_moe.experts.deepspeed_experts:
                w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=layer.weight.shape)
                layer.weight.data = torch.nn.Parameter(w)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.zero_()
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wi_1.weight.shape)
            module.wi_1.weight.data = torch.nn.Parameter(w)
        else:
            w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wi.weight.shape)
            module.wi.weight.data = torch.nn.Parameter(w)
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()

        # module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
        w = torch.normal(mean=0.0, std=factor * ((self.config.d_model) ** -0.5), size=module.wo.weight.shape)
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
        module.q.weight.data = torch.nn.Parameter(torch.normal(mean=0.0,
                                                               std=factor * ((d_model * key_value_proj_dim) ** -0.5),
                                                               size=module.q.weight.shape))
        module.k.weight.data = torch.nn.Parameter(torch.normal(mean=0.0,
                                                               std=factor * (d_model**-0.5),
                                                               size=module.k.weight.shape))
        module.v.weight.data = torch.nn.Parameter(torch.normal(mean=0.0,
                                                               std=factor * (d_model**-0.5),
                                                               size=module.v.weight.shape))
        module.o.weight.data = torch.nn.Parameter(torch.normal(mean=0.0,
                                                               std=factor * ((n_heads * key_value_proj_dim) ** -0.5),
                                                               size=module.o.weight.shape))
        if module.has_relative_attention_bias:
            w = torch.normal(mean=0.0, std=factor * ((d_model) ** -0.5), size=module.relative_attention_bias.weigh.shape)
            module.relative_attention_bias.weight.data = torch.nn.Parameter(w)


def T5ForConditionalGenerationForward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    decoder_head_mask: Optional[torch.FloatTensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )

def monkey_patching():
    # MoE monkey patching
    T5DenseReluDense.__init__ = T5DenseReluDenseInit
    T5DenseReluDense.forward = T5DenseReluDenseForward
    T5PreTrainedModel.__init_weights = InitWeights

    # T5 model monkey patch to change Loss ignore index, because its hardcode
    T5ForConditionalGeneration.forward = T5ForConditionalGenerationForward


def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_parameters(model, moe_enabled):
    if moe_enabled:
        return create_moe_param_groups(model)
    else:
        return filter(lambda p: p.requires_grad, model.parameters())


def create_model(decoder_start_token_id, dropout_rate, n_positions, num_layers, num_heads, ff_dim,
                 moe, moe_num_experts, moe_ep_size):
    config = T5Config(n_positions=n_positions, output_past=True)
    config.decoder_start_token_id = decoder_start_token_id
    config.dropout_rate = dropout_rate
    if num_layers is not None:
        config.num_layers = num_layers
    if num_heads is not None:
        config.num_heads = num_heads
    if ff_dim is not None:
        config.d_ff = ff_dim
    config.moe_enabled = moe
    config.moe_num_experts = moe_num_experts
    config.moe_ep_size = moe_ep_size
    return T5ForConditionalGeneration(config)


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
    dropout: float = 0.1,

    # Training Params
    batch_size: int = 8,
    num_iterations: int = 1000000,
    checkpoint_every: int = -1,

    # DeepSpeed Params
    fp16: bool = False,
    initial_scale_power: int = 32,
    zero_stage: int = 0,
    moe: bool = False,
    moe_num_experts: int = 128,
    moe_ep_size: int = 1,
    local_rank: int = -1
):
    deepspeed.init_distributed(dist_backend='hccl')

    if checkpoint_dir is None and load_checkpoint_dir is None:
        t5_utils.log_dist('Need to specify one of checkpoint_dir or load_checkpoint_dir',
                          ranks=[0], level=t5_utils.logging.ERROR)
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        t5_utils.log_dist('Cannot specify both checkpoint_dir and load_checkpoint_dir',
                          ranks=[0], level=t5_utils.logging.ERROR)
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
            'dropout': dropout,

            # Training Params
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'checkpoint_every': checkpoint_every,

            # DeepSpeed Params
            'fp16': fp16,
            'initial_scale_power': initial_scale_power,
            'zero_stage': zero_stage,
            'moe': moe,
            'moe_num_experts': moe_num_experts,
            'moe_ep_size': moe_ep_size
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
        dropout = hparams.get('dropout', dropout)

        # Training Params
        batch_size = hparams.get('batch_size', batch_size)
        _num_iterations = hparams.get('num_iterations', num_iterations)
        num_iterations = max(num_iterations, _num_iterations)
        checkpoint_every = hparams.get('checkpoint_every', checkpoint_every)

        # DeepSpeed Params
        fp16 = hparams.get('fp16', fp16)
        initial_scale_power = hparams.get('initial_scale_power', initial_scale_power)
        zero_stage = hparams.get('zero_stage', zero_stage)
        moe = hparams.get('moe', moe)
        moe_num_experts = hparams.get('moe_num_experts', moe_num_experts)
        moe_ep_size = hparams.get('moe_ep_size', moe_ep_size)

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
    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    t5_utils.log_dist('Dataset Creation Done with length: {}'.format(len(dataset)),
                      ranks=[0], level=t5_utils.logging.INFO)

    t5_utils.log_dist('Creating Model', ranks=[0], level=t5_utils.logging.INFO)
    model = create_model(tokenizer.added_tokens_encoder['<decoder>'], dropout, n_positions, num_layers, num_heads,
                         ff_dim, moe, moe_num_experts, moe_ep_size)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config,
                                          training_data=dataset, collate_fn=collate_fn)

    t5_utils.log_dist('DeepSpeed engine created', ranks=[0], level=t5_utils.logging.INFO)
    t5_utils.log_dist(f'Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}', ranks=[0],
                      level=t5_utils.logging.INFO)

    # Load Model checkpoint
    start_step = 0
    if load_checkpoint_dir is not None:
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        start_step = client_state['checkpoint_step']
        num_iterations -= start_step

    # The Training Loop
    model.train()
    step = start_step
    epoch_length = len(dataset) // batch_size
    last_losses = []
    for epoch in range(0, math.ceil(num_iterations / epoch_length)):
        if step == start_step:
            start_train_loader = step % epoch_length
        else:
            start_train_loader = 0

        for i, batch in enumerate(train_loader, start_train_loader):
            start_time = datetime.datetime.now()
            loss = model(input_ids=batch[0].to(torch.npu.current_device()),
                         labels=batch[1].to(torch.npu.current_device()),
                         attention_mask=batch[2].to(torch.npu.current_device()),
                         decoder_attention_mask=batch[3].to(torch.npu.current_device())).loss
            model.backward(loss)
            model.step()

            t5_utils.log_dist('Step: {}\tDuration: {}\t Training Loss: {}'
                              .format(step, datetime.datetime.now()-start_time, loss.item()),
                              ranks=[0], level=t5_utils.logging.INFO)

            if step != 0 and checkpoint_every > 0 and step % checkpoint_every == 0:
                model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step},
                                      tag='step_{}'.format(step))
                t5_utils.log_dist("Saved model to {0}".format(exp_dir), ranks=[0], level=t5_utils.logging.INFO)

            step += 1
            if step > num_iterations - 1000:
                last_losses.append(loss)
            if step >= num_iterations:
                break

    # Save the last checkpoint if not saved yet
    if checkpoint_every > 0 and step % checkpoint_every != 0:
        model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step}, tag='step_{}'.format(step))
        t5_utils.log_dist("Saved model to {0}".format(exp_dir), ranks=[0], level=t5_utils.logging.INFO)

    t5_utils.log_dist('finished training, last {} losses average: {}'
                      .format(len(last_losses), sum(last_losses) / len(last_losses)),
                      ranks=[0], level=t5_utils.logging.INFO)


if __name__ == '__main__':
    option = {}
    option["ACL_OP_COMPILER_CACHE_MODE"] = "enable"  # cache功能启用
    option["ACL_OP_COMPILER_CACHE_DIR"] = "./cache"  # cache所在文件夹

    print("option:", option)
    torch.npu.set_option(option)
    transformers.set_seed(123)
    torch.npu.manual_seed_all(123)
    monkey_patching()
    fire.Fire(train)