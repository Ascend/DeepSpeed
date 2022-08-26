from functools import wraps
import sys
import torch
from torch import Tensor
import torch.distributed as dist
from typing import Any, Tuple, cast
from deepspeed.moe import sharded_moe

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