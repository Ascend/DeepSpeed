import torch
import torch_npu
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.fp16 import loss_scaler, unfused_optimizer
from . import FLAG_SUPPORT_INF_NAN


# loss_scaler============
def backward(self, loss, retain_graph=False):
    if not FLAG_SUPPORT_INF_NAN:
        clear_npu_overflow_flag()
    scaled_loss = loss * self.loss_scale
    scaled_loss.backward(retain_graph=retain_graph)


def has_overflow_serial(self, params):
    if not FLAG_SUPPORT_INF_NAN:
        grads = [p.grad.data for p in params if p.grad is not None]
        return torch_npu.npu.utils.npu_check_overflow(grads)

    for p in params:
        if p.grad is not None and self._has_inf_or_nan(p.grad.data):
            return True
    return False


loss_scaler.LossScalerBase.backward = backward
loss_scaler.DynamicLossScaler.has_overflow_serial = has_overflow_serial
