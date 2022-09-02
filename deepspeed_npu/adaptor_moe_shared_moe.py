from functools import wraps
import sys
import torch
import deepspeed
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Tuple, cast, Optional
from deepspeed.moe import sharded_moe
from deepspeed.moe.sharded_moe import gumbel_rsample, _capacity, einsum, \
    exp_selection_uniform_map, _top_idx, _one_hot_to_float, TUTEL_INSTALLED

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        # ASCEND AVOID
        input = input.contiguous()
        if input.dim() == 4:
            input = input.npu_format_cast(0)
        elif input.dim() == 5:
            input = input.npu_format_cast(30)
        else:
            input = input.npu_format_cast(2)
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

sharded_moe._ALLToALL = _AllToAll
for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, '_AllToAll'):
        setattr(v, '_AllToAll', _AllToAll)

def warning_once(fn):
    fn.warned = False
    def wrapper(*args, **kwargs):
        if not fn.warned:
            fn.warned = True
            return fn(*args, **kwargs)
    return wrapper

def print_warning():
    print('[warning] torch.jit.script is disabled in this version...')

def empty_jit_wrapper(fn):
    print_warning()
    return fn

torch.jit.script = empty_jit_wrapper


def one_hot_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

torch.nn.functional.one_hot = one_hot_wrapper(torch.nn.functional.one_hot)


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               use_tutel: bool = False) -> Tuple[Tensor,
                                                 Tensor,
                                                 Tensor,
                                                 Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates,
                         torch.tensor(capacity_factor),
                         torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    num_experts = int(gates.shape[1])
    # ASCEND AVOID
    mask1 = F.one_hot(indices1_s.int(), num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        capacity = new_capacity

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0,
                                 device=logits.device).cpu(),
                high=torch.tensor(1.0,
                                  device=logits.device).cpu()).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape).npu()
    else:
        mask1_rand = mask1

    assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [indices1_s,], [locations1_s,], [gates1_s,], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

deepspeed.moe.sharded_moe.top1gating = top1gating
