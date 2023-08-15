from typing import List
import torch
import torch_npu
from torch import Tensor

import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.zero.mics_utils import MiCS_CommGroups
from deepspeed.accelerator import get_accelerator


def allreduce_mics_shard_grads(self, params, partitioned_grads_buffers: List[Tensor]):
    """
    """
    # TODO: improve the condition check
    if not self.is_gradient_accumulation_boundary or \
        len(partitioned_grads_buffers) == 0:
        return

    mics_comm_groups: MiCS_CommGroups = params[0].comm
    param_repli_group = mics_comm_groups.param_repli_group
    param_repli_size = mics_comm_groups.param_repli_size

    if param_repli_size is None or param_repli_size <= 1:
        return
    if not get_accelerator().on_accelerator(partitioned_grads_buffers[0]):
        raise RuntimeError("Local sharding has no support for CPU offloading")

    # manually coalescing all-reduce
    # to work around HCCL limitation https://gitee.com/ascend/pytorch/issues/I7I7YL
    aggregated_buffer: Tensor = torch.cat(partitioned_grads_buffers)
    aggregated_buffer.div_(param_repli_size)
    dist.all_reduce(aggregated_buffer, group=param_repli_group)
    offset = 0
    for grad_buff in partitioned_grads_buffers:
        grad_buff.view(-1).copy_(aggregated_buffer.narrow(0, offset, grad_buff.numel()))
        offset += grad_buff.numel()


deepspeed.runtime.zero.mics.MiCS_Optimizer.allreduce_mics_shard_grads = allreduce_mics_shard_grads
