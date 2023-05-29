import torch
import deepspeed
from torch_npu.npu import clear_npu_overflow_flag


def backward(self, loss, retain_graph=False):
    clear_npu_overflow_flag()
    scaled_loss = loss * self.loss_scale
    scaled_loss.backward(retain_graph=retain_graph)


def has_overflow_serial(self, params):
    grads = [p.grad.data for p in params if p.grad is not None]
    return torch_npu._amp_foreach_non_finite_check_(grads)


deepspeed.runtime.fp16.loss_scaler.LossScalerBase.backward = backward
deepspeed.runtime.fp16.loss_scaler.DynamicLossScaler.has_overflow_serial = has_overflow_serial
