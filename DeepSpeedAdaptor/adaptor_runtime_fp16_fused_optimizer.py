from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.fp16 import fused_optimizer
# fused_optimizer============
def fused_optimizer_backward(self, loss, create_graph=False, retain_graph=False):
    """
    :attr:`backward` performs the following steps:

    1. fp32_loss = loss.float()
    2. scaled_loss = fp32_loss*loss_scale
    3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
    """
    clear_npu_overflow_flag()
    scaled_loss = (loss.float()) * self.cur_scale

    scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

fused_optimizer.FP16_Optimizer.backward = fused_optimizer_backward