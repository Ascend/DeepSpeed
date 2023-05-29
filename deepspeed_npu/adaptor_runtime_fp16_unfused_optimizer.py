import sys
import torch
import deepspeed
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.utils import logger
from deepspeed.runtime.utils import get_global_norm, CheckOverflow, get_weight_norm
from deepspeed.moe.utils import split_params_grads_into_shared_and_expert_params


def step(self, closure=None):
    # if self.fused_lamb_legacy:
    #     return self.step_fused_lamb()

    self.overflow = self.overflow_checker.check()
    prev_scale = self.cur_scale

    self._update_scale(self.overflow)
    if self.overflow:
        if self.verbose:
            logger.info("[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                        "scale: {}, reducing to {}".format(prev_scale, self.cur_scale))
        return self.overflow

    norm_groups = []
    for i, group in enumerate(self.fp16_groups):
        grads_for_norm, _ = split_params_grads_into_shared_and_expert_params(group)
        norm_group_value = 0.0
        if len(grads_for_norm) > 0:
            norm_group_value = get_weight_norm(grads_for_norm, mpu=self.mpu)
        norm_groups.append(norm_group_value)

        # copying gradients to fp32 to wor  k with fp32 parameters
        for fp32_param, fp16_param in zip(self.fp32_groups[i], self.fp16_groups[i]):
            if fp16_param.grad is None:
                fp32_param.grad = torch.zeros(fp16_param.size(), dtype=fp32_param.dtype, device=fp32_param.device)
            else:
                fp32_param.grad = fp16_param.grad.to(fp32_param.dtype)

    self._global_grad_norm = get_global_norm(norm_list=norm_groups)
    self.unscale_and_clip_grads(self._global_grad_norm)

    self.optimizer.step()

    for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
        for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):

            #remove the fp32 grad
            fp32_param.grad = None

            #copy data from fp32 to fp16
            fp16_param.data.copy_(fp32_param.data)

    return self.overflow


def backward(self, loss, create_graph=False, retain_graph=False):
    clear_npu_overflow_flag()
    if self.custom_loss_scaler:
        scaled_loss = self.external_loss_scale * loss
        scaled_loss.backward()
    else:
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)


deepspeed.runtime.fp16.unfused_optimizer.FP16_UnfusedOptimizer.step = step
deepspeed.runtime.fp16.unfused_optimizer.FP16_UnfusedOptimizer.backward = backward
