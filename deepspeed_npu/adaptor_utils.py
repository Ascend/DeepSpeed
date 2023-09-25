import torch
import torch_npu

from functools import wraps
from deepspeed import comm as dist
import DeepSpeed

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


def wrapper_clone(fn):
    @wraps(fn)
    def clone_tensors_for_torch_save(*args, **kwargs):
        """
        Returns a copy of ``item`` with all enclosed tensors replaced by clones on a specified device.
        Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

        Parameters:
            - ``item``: tensor to clone or (possibly nested) container of tensors to clone.
            - ``device``: target device (defaults to 'cpu')

        Returns:
            - copy of ``item`` with cloned tensors on target device
        """
        if torch.is_tensor(args[0]):
            return args[0].detach().clone().to(args[1])
        elif isinstance(args[0], list):
            return [clone_tensors_for_torch_save(v, args[1]) for v in args[0]]
        elif isinstance(args[0], tuple):
            return tuple([clone_tensors_for_torch_save(v, args[1]) for v in args[0]])
        elif isinstance(args[0], dict):
            return type(args[0])({k: clone_tensors_for_torch_save(v, args[1]) for k, v in args[0].items()})
        else:
            return args[0]

torch.cuda.nvtx = torch.ones
dist.send = wrapper_dist_send(dist.send)
dist.recv = wrapper_dist(dist.recv)
dist.all_reduce = wrapper_dist(dist.all_reduce)
DeepSpeed.deepspeed.checkpoint.utils.clone_tensors_for_torch_save = clone_tensors_for_torch_save(DeepSpeed.deepspeed.checkpoint.utils.clone_tensors_for_torch_save):

