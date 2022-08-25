import torch
from torch import Tensor
from deepspeed.runtime.zero import partition_parameters
from deepspeed.utils import instrument_w_nvtx
def torch_allgather_fn(input_tensor: Tensor, output_tensor: Tensor, group):
    output_tensors = list(
        torch.chunk(output_tensor,
                    torch.distributed.get_world_size(group)))
    # ASCEND AVOID
    new_output_tensors = [x.clone() for x in output_tensors]
    instrument_w_nvtx(torch.distributed.all_gather)(
        new_output_tensors,
        input_tensor.clone(),
        # input_tensor,
        group=group,
        async_op=False,
    )
    for i in range(len(new_output_tensors)):
        output_tensors[i].copy_(new_output_tensors[i])
    return torch.distributed.barrier(async_op=True)

partition_parameters.torch_allgather_fn = torch_allgather_fn