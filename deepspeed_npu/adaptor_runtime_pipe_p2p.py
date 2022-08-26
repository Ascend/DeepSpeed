import torch.distributed as dist
from deepspeed.runtime.pipe.p2p import can_send_recv, _is_valid_send_recv, _get_send_recv_group, _async, _grid

def send(tensor, dest_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesn't support async_op true"
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    if async_op:
        global _async
        op = dist.isend(tensor, dest_rank)
        _async.append(op)
    else:
        if can_send_recv():
            if tensor.dim() == 4:
                tensor = tensor.npu_format_cast(0)
            elif tensor.dim() == 5:
                tensor = tensor.npu_format_cast(30)
            else:
                tensor = tensor.npu_format_cast(2)
            return dist.send(tensor, dest_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            src_rank = _grid.stage_to_global(stage_id=src_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)

for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'send'):
        setattr(v, 'send', send)