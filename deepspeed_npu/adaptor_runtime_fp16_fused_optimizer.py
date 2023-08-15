import sys
import torch
import deepspeed
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.utils import get_global_norm, get_grad_norm
from deepspeed.utils import groups, logger, log_dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.runtime.utils import get_global_norm, CheckOverflow, get_weight_norm
from . import FLAG_SUPPORT_INF_NAN


def step(self, closure=None):
    """
    Not supporting closure.
    """

    # if self.fused_adam_legacy:
    #     return self.step_fused_adam()

    COMPUTE_NORM = "compute_norm"
    OVERFLOW_CHECK = 'overflow_check'
    OVERFLOW_TIMERS = [COMPUTE_NORM, OVERFLOW_CHECK]
    UNSCALE_AND_CLIP = 'unscale_and_clip'
    BASIC_STEP = 'basic_step'
    UPDATE_FP16 = 'update_fp16'
    STEP_TIMERS = OVERFLOW_TIMERS + [UNSCALE_AND_CLIP, BASIC_STEP, UPDATE_FP16]

    # First determine if there is overflow.
    self.start_timers([OVERFLOW_CHECK])
    fp16_params = []
    for i, group in enumerate(self.fp16_groups):
        fp16_params.extend([p for p in group if p.grad is not None])
    self.overflow = self.overflow_checker.has_overflow(fp16_params)
    self.stop_timers([OVERFLOW_CHECK])
    prev_scale = self.cur_scale
    self._update_scale(self.overflow)
    if self.overflow:
        if self.verbose:
            log_dist(
                "Overflow detected. Skipping step. Attempted loss "
                f"scale: {prev_scale}, reducing to {self.cur_scale}",
                ranks=[0])
        # Clear gradients
        for i, group in enumerate(self.fp16_groups):
            for p in group:
                p.grad = None

        self.log_timers(OVERFLOW_TIMERS)
        return self.overflow

    grads_groups_flat = []
    for i, group in enumerate(self.fp16_groups):
        data_type = self.fp32_groups_flat[i].dtype

        grads_groups_flat.append(
            _flatten_dense_tensors([
                torch.zeros(p.size(), dtype=data_type, device=p.device) if p.grad is None else p.grad.to(data_type)
                for p in group
            ]))

        for p in group:
            p.grad = None

        self.fp32_groups_flat[i].grad = grads_groups_flat[i]

    self.start_timers([COMPUTE_NORM])

    all_groups_norm = get_grad_norm(self.fp32_groups_flat, mpu=self.mpu)

    self.stop_timers([COMPUTE_NORM])

    if self.has_moe_layers:
        all_groups_norm = self._get_norm_with_moe_layers(all_groups_norm)

    scaled_global_grad_norm = get_global_norm(norm_list=[all_groups_norm])

    # Stash unscaled gradient norm
    self._global_grad_norm = scaled_global_grad_norm / self.cur_scale

    self.start_timers([UNSCALE_AND_CLIP])
    self.unscale_and_clip_grads(grads_groups_flat, scaled_global_grad_norm)
    self.stop_timers([UNSCALE_AND_CLIP])

    self.start_timers([BASIC_STEP])
    self.optimizer.step()
    self.stop_timers([BASIC_STEP])

    #get rid of the fp32 gradients. Not needed anymore
    for group in self.fp32_groups_flat:
        group.grad = None

    self.start_timers([UPDATE_FP16])

    for i in range(len(self.fp16_groups)):
        updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
        for p, q in zip(self.fp16_groups[i], updated_params):
            p.data.copy_(q.data)

    self.stop_timers([UPDATE_FP16])

    self.log_timers(STEP_TIMERS)

    self.step_count += 1

    return self.overflow


def backward(self, loss, create_graph=False, retain_graph=False):
    if not FLAG_SUPPORT_INF_NAN:
        clear_npu_overflow_flag()
    if self.custom_loss_scaler:
        scaled_loss = self.external_loss_scale * loss
        scaled_loss.backward()
    else:
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)


deepspeed.runtime.fp16.fused_optimizer.FP16_Optimizer.step = step
deepspeed.runtime.fp16.fused_optimizer.FP16_Optimizer.backward = backward