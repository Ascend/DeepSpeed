import sys
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.fp16 import unfused_optimizer
from . import FLAG_SUPPORT_INF_NAN


# unfused_optimizer============
class Fp16UnfusedOptimizerNpu(unfused_optimizer.FP16_UnfusedOptimizer):
    def __init__(self,
                 init_optimizer,
                 deepspeed=None,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 clip_grad=0.0,
                 fused_lamb_legacy=False):
        super().__init__(init_optimizer,
                         deepspeed,
                         static_loss_scale,
                         dynamic_loss_scale,
                         dynamic_loss_args,
                         verbose,
                         mpu,
                         clip_grad,
                         fused_lamb_legacy)

    def step(self, closure=None):
        self.fused_lamb_legacy = False
        super().step(closure)

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if not FLAG_SUPPORT_INF_NAN:
            clear_npu_overflow_flag()
        scaled_loss = (loss.float()) * self.cur_scale

        scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)


unfused_optimizer.FP16_UnfusedOptimizer = Fp16UnfusedOptimizerNpu
for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'FP16_UnfusedOptimizer'):
        setattr(v, 'FP16_UnfusedOptimizer', Fp16UnfusedOptimizerNpu)
