from functools import wraps
import copy
import torch
import torch.distributed as dist
import torch_npu
torch.cuda.nvtx = torch.ones

# recv/all_reduce operations need modify the inputs, copy back is required
def wrapper_dist(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and args[0].dtype == torch.long:
            new_args = copy.deepcopy(list(args))
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

dist.send = wrapper_dist_send(dist.send)
dist.recv = wrapper_dist(dist.recv)
dist.all_reduce = wrapper_dist(dist.all_reduce)
