from functools import wraps
import torch
import torch.distributed as dist
torch.cuda.nvtx = torch.ones

def wrapper_dist(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0].dtype == torch.long:
            new_args = [args[0].int()]
            fn(*new_args, **kwargs)
            args[0].copy_(new_args[0].long())
            return
        fn(*args, **kwargs)
    
    return wrapper

dist.send = wrapper_dist(dist.send)
dist.recv = wrapper_dist(dist.recv)
dist.all_reduce = wrapper_dist(dist.all_reduce)
