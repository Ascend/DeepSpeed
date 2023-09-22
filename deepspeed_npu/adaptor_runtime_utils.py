import sys
import torch
import torch_npu
import deepspeed

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank, is_model_parallel_parameter
from . import FLAG_SUPPORT_INF_NAN

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf


def check_using_norm(self, norm_group, reduce_overflow=True):
    # TODO: I don't think reduce_overflow is needed if mpu is None
    overflow = -1 in norm_group
    overflow_npu = get_accelerator().FloatTensor([overflow])
    if self.has_moe_params:
        # In this case, we need to do an all_reduce across
        # the expert_parallel_group, so that if there was
        # an overflow due to expert weights, we detect it

        # Only need to check groups.get_largest_expert_parallel_group()
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
    if self.mpu is not None:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.mpu.get_data_parallel_group())
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
    elif reduce_overflow:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX)
        dist.barrier()
    overflow = overflow_npu[0].item()
    return bool(overflow)


def has_overflow_serial(self, params):
    if not FLAG_SUPPORT_INF_NAN:
        grads = [p.grad.data for p in params if p.grad is not None]
        return torch_npu.npu_check_overflow(grads)

    for i, p in enumerate(params):
        if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
            return True
    return False


def has_overflow(self, params, has_moe_params=None):
    if has_moe_params is None:
        has_moe_params = self.has_moe_params

    overflow = self.has_overflow_serial(params)
    # Since each model parallel GPU carries only part of the model,
    # make sure overflow flag is synced across all the model parallel GPUs
    overflow_npu = get_accelerator().IntTensor([overflow])
    # deepspeeed.comm.all_reduce(overflow_npu,
    #                             op=deepspeed.comm.ReduceOp.MAX,
    #                             group=mpu.get_model_parallel_group())
    if has_moe_params:
        # All reduce this across expert_parallel_group, so that if an expert
        # overflows, we detect it here
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
    if self.zero_reduce_scatter:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=dist.get_world_group())
    elif self.mpu is not None:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.mpu.get_data_parallel_group())
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
    else:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=dist.get_world_group())

    overflow = overflow_npu[0].item()
    return bool(overflow)


def get_grad_norm(parameters, norm_type=2, mpu=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_npu, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_npu, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow = False
    if not FLAG_SUPPORT_INF_NAN:
        overflow = torch_npu.npu_check_overflow([total_norm_npu])
        if mpu is not None:
            overflow_npu = torch.npu.IntTensor([overflow])
            dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            overflow = overflow_npu.item()

    if overflow or total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def get_weight_norm(parameters, norm_type=2, mpu=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_npu, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_npu, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow = False
    if not FLAG_SUPPORT_INF_NAN:
        overflow = torch_npu.npu_check_overflow([total_norm_npu])
        if mpu is not None:
            overflow_npu = torch.npu.IntTensor([overflow])
            dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            overflow = overflow_npu.item()

    if overflow or total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


deepspeed.runtime.utils.CheckOverflow.check_using_norm = check_using_norm
deepspeed.runtime.utils.CheckOverflow.has_overflow_serial = has_overflow_serial
deepspeed.runtime.utils.CheckOverflow.has_overflow = has_overflow
deepspeed.runtime.utils.CheckOverflow.get_grad_norm = get_grad_norm
deepspeed.runtime.utils.CheckOverflow.get_weight_norm = get_weight_norm