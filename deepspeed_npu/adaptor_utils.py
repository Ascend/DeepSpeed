import torch
import torch_npu

from functools import wraps
from deepspeed import comm as dist
from deepspeed.checkpoint import utils

# recv/all_reduce operations need modify the inputs, copy back is required
def wrapper_dist(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and args[0].dtype == torch.long:
            new_args = list(args)
            new_args[0] = new_args[0].int()
            tmp = fn(*new_args, **kwargs)
            args[0].copy_(new_args[0].long())
            return tmp
        return fn(*args, **kwargs)

    return wrapper


def wrapper_dist_send(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = list(args)
        if args[0].dtype == torch.long:
            args[0] = args[0].int()

        if args[0].dim() == 4:
            args[0] = torch_npu.npu_format_cast(args[0], 0)
        elif args[0].dim() == 5:
            args[0] = torch_npu.npu_format_cast(args[0], 30)
        else:
            args[0] = torch_npu.npu_format_cast(args[0], 2)
        return fn(*args, **kwargs)

    return wrapper


def clone_tensors_for_torch_save(item, device=torch.device('cpu')):
    """
    Returns a copy of ``item`` with all enclosed tensors replaced by clones on a specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

    Parameters:
        - ``item``: tensor to clone or (possibly nested) container of tensors to clone.
        - ``device``: target device (defaults to 'cpu')

    Returns:
        - copy of ``item`` with cloned tensors on target device
    """
    if torch.is_tensor(item):
        return item.detach().clone().to(device)
    elif isinstance(item, list):
        return [clone_tensors_for_torch_save(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([clone_tensors_for_torch_save(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: clone_tensors_for_torch_save(v, device) for k, v in item.items()})
    else:
        return item
torch.cuda.nvtx = torch.ones
dist.send = wrapper_dist_send(dist.send)
dist.recv = wrapper_dist(dist.recv)
dist.all_reduce = wrapper_dist(dist.all_reduce)
utils.clone_tensors_for_torch_save = clone_tensors_for_torch_save

