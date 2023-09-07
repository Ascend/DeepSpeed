import sys
import torch
import torch_npu
import torch.distributed as dist
try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
from deepspeed.runtime import utils
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank
from deepspeed.utils import groups
from deepspeed.runtime.utils import is_model_parallel_parameter
import deepspeed_npu.adaptor_runtime_activation_checkpointing_checkpointing as checkpointing

def check_using_norm(self, norm_group, reduce_overflow=True):
    # TODO: I don't think reduce_overflow is needed if mpu is None
    overflow = -1 in norm_group
    overflow_npu = torch.npu.FloatTensor([overflow])
    if self.has_moe_params:
        # In this case, we need to do an all_reduce across
        # the expert_parallel_group, so that if there was
        # an overflow due to expert weights, we detect it

        # Only need to check groups.get_largest_expert_parallel_group()
        dist.all_reduce(overflow_npu,
                        op=dist.ReduceOp.MAX,
                        group=groups._get_max_expert_parallel_group())
    if self.mpu is not None:
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=self.mpu.get_data_parallel_group())
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=self.mpu.get_model_parallel_group())
    elif reduce_overflow:
        dist.all_reduce(overflow_npu, op=torch.distributed.ReduceOp.MAX)
        dist.barrier()
    overflow = overflow_npu[0].item()
    return bool(overflow)

def has_overflow_serial(self, params):
    grads = [p.grad.data for p in params if p.grad is not None]
    if torch_npu.__version__ >= "2.1":
        res = torch_npu._amp_foreach_non_finite_check(grads)
    else:
        res = torch_npu._amp_foreach_non_finite_check_(grads)
    return res

def has_overflow(self, params, has_moe_params=None):
    if has_moe_params is None:
        has_moe_params = self.has_moe_params
    overflow = self.has_overflow_serial(params)
    # Since each model parallel GPU carries only part of the model,
    # make sure overflow flag is synced across all the model parallel GPUs
    overflow_npu = torch.npu.IntTensor([overflow])
    # torch.distributed.all_reduce(overflow_npu,
    #                             op=torch.distributed.ReduceOp.MAX,
    #                             group=mpu.get_model_parallel_group())
    if has_moe_params:
        # All reduce this across expert_parallel_group, so that if an expert
        # overflows, we detect it here
        dist.all_reduce(overflow_npu,
                        op=dist.ReduceOp.MAX,
                        group=groups._get_max_expert_parallel_group())
    if self.zero_reduce_scatter:
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=torch.distributed.group.WORLD)
    elif self.mpu is not None:
        # ASCEND AVOID OVERFLOW
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=self.mpu.get_data_parallel_group())
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=self.mpu.get_model_parallel_group())
    elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
        torch.distributed.all_reduce(overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=torch.distributed.group.WORLD)
    else:
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=torch.distributed.group.WORLD)

    overflow = overflow_npu[0].item()
    return bool(overflow)

utils.CheckOverflow.check_using_norm = check_using_norm
utils.CheckOverflow.has_overflow_serial = has_overflow_serial
utils.CheckOverflow.has_overflow = has_overflow

def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
        if mpu is not None:
            # ASCEND VOID OVERFLOW
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow = torch_npu._amp_foreach_non_finite_check_([total_norm_npu])
    if mpu is not None:
        overflow_npu = torch.npu.IntTensor([overflow])
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        overflow = overflow_npu.item()

    if overflow or total_norm == float('inf') or \
        total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow = torch_npu._amp_foreach_non_finite_check_([total_norm_npu])
    if mpu is not None:
        overflow_npu = torch.npu.IntTensor([overflow])
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        overflow = overflow_npu.item()

    if overflow or total_norm == float('inf') or \
        total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'get_grad_norm'):
        setattr(v, 'get_grad_norm', get_grad_norm)

for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'get_weight_norm'):
        setattr(v, 'get_weight_norm', get_weight_norm)
