import torch
import torch_npu
import deepspeed
if torch_npu.__version__ >= "2.1":
    from torch.distributed.distributed_c10d import _get_global_rank
else:
    from torch_npu.distributed.distributed_c10d import _get_global_rank
from deepspeed.utils.groups import _get_data_parallel_group, _get_expert_data_parallel_group


def _get_broadcast_src_rank():
    return _get_global_rank(_get_data_parallel_group(), 0)


def _get_expert_broadcast_src_rank(group_name):
    return _get_global_rank(_get_expert_data_parallel_group(group_name), 0)


deepspeed.utils.groups._get_broadcast_src_rank = _get_broadcast_src_rank
deepspeed.utils.groups._get_expert_broadcast_src_rank = _get_expert_broadcast_src_rank
