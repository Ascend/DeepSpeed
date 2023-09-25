import torch
from torch import Tensor
from deepspeed.runtime.comm import coalesced_collectives


def torch_reduce_scatter_fn(input_tensor: Tensor, output_tensor: Tensor, group):
    input_tensor_lst = list(
        torch.chunk(input_tensor,
                    torch.distributed.get_world_size(group)))
    # ASCEND AVOID
    new_input_tensor_lst = [x.clone() for x in input_tensor_lst]
    new_output_tensor = output_tensor.clone()
    torch.distributed.reduce_scatter(
        new_output_tensor,
        new_input_tensor_lst,
        group=group,
    )
    output_tensor.copy_(new_output_tensor)


coalesced_collectives.torch_reduce_scatter_fn = torch_reduce_scatter_fn
