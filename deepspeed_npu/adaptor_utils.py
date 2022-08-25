from functools import wraps
import torch
import torch.distributed as dist
torch.cuda.nvtx = torch.ones

def wrapper_send_recv(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        new_args = list(args)
        if args[0].dtype == torch.long:
            new_args[0] = torch.int
        args = new_args
        return fn(*args, **kwargs)
    
    return wrapper

dist.send = wrapper_send_recv(dist.send)
dist.recv = wrapper_send_recv(dist.recv)
