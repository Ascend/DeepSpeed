import torch
import torch_npu
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.fp16 import fused_optimizer
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE
from deepspeed.runtime.utils import get_global_norm, CheckOverflow, get_weight_norm
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT
from deepspeed.utils import logger
from apex.contrib.combine_tensors import combine_npu
from deepspeed_npu.adaptor_ops_adam_fused_adam import FusedAdamNPU


# fused_optimizer============
def FP16OptimizerInit(
        self,
        init_optimizer,
        deepspeed=None,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        initial_dynamic_scale=2 ** 32,
        dynamic_loss_args=None,
        verbose=True,
        mpu=None,
        clip_grad=0.0,
        fused_adam_legacy=False,
        timers=None
):
    self.fused_adam_legacy = fused_adam_legacy
    self.timers = timers
    self.deepspeed = deepspeed
    self.using_pipeline = self.deepspeed.pipeline_parallelism

    self._global_grad_norm = 0.

    if not torch.npu.is_available:
        raise SystemError("Cannot use fp16 without NPU.")
    self.optimizer = init_optimizer

    self.is_combined = False

    # param flattened by groups
    self.fp16_groups = []
    self.fp32_groups = []

    self.fp16_params_combine_groups = []
    self.fp16_grads_combine_groups = []

    self.fp32_params_combine_groups = []
    self.fp32_grads_combine_groups = []

    self.shared_grads_combine_groups = []

    # loop to deal with groups
    for i, param_group in enumerate(self.optimizer.param_groups):
        # fp16 weights that represents the actual model weights
        self.fp16_groups.append(param_group['params'])

        # creating a fp32 copy of the weights that will be updated first then copied to fp16 weights
        fp32_group = [p.data.float() for p in param_group['params']]

        for p in fp32_group:
            p.requires_grad = True

        self.fp32_groups.append(fp32_group)
        param_group['params'] = self.fp32_groups[i]

    # we may have a way of fusing dynamic scale. Do not support for now
    if dynamic_loss_scale:
        self.dynamic_loss_scale = True
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = 2

        if dynamic_loss_args is None:
            self.cur_scale = initial_dynamic_scale
            self.scale_window = 1000
            self.min_loss_scale = 1
        else:
            self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
            self.scale_window = dynamic_loss_args[SCALE_WINDOW]
            self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
    else:
        self.dynamic_loss_scale = False
        self.cur_iter = 0
        self.cur_scale = static_loss_scale

    self.verbose = verbose

    self.clip_grad = clip_grad
    self.norm_type = 2

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm
    else:
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # model parallel object
    self.mpu = mpu

    self.overflow = False
    self.overflow_checker = CheckOverflow(self.fp16_groups,
                                          mpu=self.mpu,
                                          deepspeed=deepspeed)
    self.initialize_optimizer_states()


def initialize_optimizer_states(self):
    for i, group in enumerate(self.fp16_groups):
        for param in group:
            param.grad = torch.zeros_like(param)

    for i, group in enumerate(self.fp32_groups):
        for param in group:
            param.grad = torch.zeros_like(param)

    self.optimizer.step()

    for i, group in enumerate(self.fp16_groups):
        for param in group:
            param.grad.data.zero_()

    for i, group in enumerate(self.fp32_groups):
        for param in group:
            param.grad.data.zero_()


def Fp16OptimizerBackward(self, loss, create_graph=False, retain_graph=False):
    torch_npu.npu_clear_float_status(torch.zeros(8, device=torch.npu.current_device()))
    scaled_loss = (loss.float()) * self.cur_scale
    scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)


def Fp16OptimizerStep(self, closure=None):
    if not self.is_combined:
        for i, group in enumerate(self.fp16_groups):
            tmp_param = []
            tmp_grad = []
            for param in group:
                tmp_param.append(param.data)
                tmp_grad.append(param.grad.data)

            self.fp16_params_combine_groups.append(combine_npu(tmp_param))
            self.fp16_grads_combine_groups.append(combine_npu(tmp_grad))

        for i, group in enumerate(self.fp32_groups):
            if not isinstance(self.optimizer, FusedAdamNPU):
                tmp_param = []
                tmp_grad = []
                for param in group:
                    tmp_param.append(param.data)
                    tmp_grad.append(param.grad.data)

                self.fp32_params_combine_groups.append(combine_npu(tmp_param))
                self.fp32_grads_combine_groups.append(combine_npu(tmp_grad))
            else:
                # fp32 wad combined in optimizer, if you do combination again,
                # the data ptr will be changed, so just get from optimizer
                self.fp32_params_combine_groups.append(self.optimizer.combined_states[i]['params'])
                self.fp32_grads_combine_groups.append(self.optimizer.combined_states[i]['grads'])

        self.is_combined = True

    # NPU overflow flag is global flag, so is equal giving params list or giving empty list
    self.overflow = self.overflow_checker.has_overflow([])
    prev_scale = self.cur_scale

    self._update_scale(self.overflow)
    if self.overflow:
        if self.verbose:
            logger.info(
                "[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                "scale: {}, reducing to {}".format(prev_scale, self.cur_scale))

            for group in self.fp16_grads_combine_groups:
                group.data.detach_()
                group.data.zero_()
        return self.overflow

    norm_groups = []
    for i, group in enumerate(self.fp16_groups):
        # grads_for_norm, _ = split_params_grads_into_shared_and_expert_params(group)
        norm_group_value = 0.0
        if len(self.fp32_params_combine_groups[i]) > 0:
            norm_group_value = self.get_combine_weight_norm(self.fp16_params_combine_groups[i])
        norm_groups.append(norm_group_value)

        self.fp32_grads_combine_groups[i].data.copy_(
            self.fp16_grads_combine_groups[i].data.to(self.fp32_grads_combine_groups[i].dtype)
        )

    self._global_grad_norm = get_global_norm(norm_list=norm_groups)
    self.unscale_and_clip_grads(self._global_grad_norm)

    self.optimizer.step()

    for i, group in self.fp32_params_combine_groups:
        self.fp32_grads_combine_groups[i].data.zero_()
        self.fp16_params_combine_groups[i].data.copy_(group.data)

    return self.overflow


def zero_grad(self, set_grads_to_None=True):
    for group in self.fp16_grads_combine_groups:
        group.data.detach_()
        group.data.zero_()


def unscale_and_clip_grads(self, total_norm, apply_scale=True):
    # compute combined scale factor for this group
    combined_scale = self.cur_scale
    if self.clip_grad > 0.:
        # norm is in fact norm*scale
        clip = ((total_norm / self.cur_scale) + 1e-6) / self.clip_grad
        if clip > 1:
            combined_scale = clip * self.cur_scale

    if apply_scale:
        for group in self.fp32_grads_combine_groups:
            group.data.mul_(1. / combined_scale)

    return combined_scale


def get_combine_weight_norm(parameters, norm_type=2, mpu=None):
    total_norm = 0.

    param_norm = parameters.data.float().norm(norm_type)
    total_norm += param_norm**norm_type

    # Sum across all model parallel GPUs.
    total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
    if mpu is not None:
        torch.distributed.all_reduce(total_norm_npu,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow = torch._amp_foreach_non_finite_check_([total_norm_npu])

    if overflow or total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def state_dict(self):
    state_dict = {}
    state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
    state_dict['cur_scale'] = self.cur_scale
    state_dict['cur_iter'] = self.cur_iter
    if state_dict['dynamic_loss_scale']:
        state_dict['last_overflow_iter'] = self.last_overflow_iter
        state_dict['scale_factor'] = self.scale_factor
        state_dict['scale_window'] = self.scale_window
    state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
    state_dict['fp32_groups'] = self.fp32_groups
    state_dict['clip_grad'] = self.clip_grad
    return state_dict


def load_state_dict(self, state_dict, load_optimizer_states=True):
    # I think it should actually be ok to reload the optimizer before the model.
    self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
    self.cur_scale = state_dict['cur_scale']
    self.cur_iter = state_dict['cur_iter']
    if state_dict['dynamic_loss_scale']:
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']
    if load_optimizer_states:
        self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
    self.clip_grad = state_dict['clip_grad']

    for current_group, saved_group in zip(self.fp32_groups, state_dict['fp32_groups']):
        for current, saved in zip(current_group, saved_group):
            current.data.copy_(saved.data)


def refresh_fp32_params(self):
    for current_group, saved_group in zip(self.fp32_groups, self.fp16_groups):
        for current, saved in zip(current_group, saved_group):
            current.data.copy_(saved.data)


fused_optimizer.FP16_Optimizer.__init__ = FP16OptimizerInit
fused_optimizer.FP16_Optimizer.initialize_optimizer_states = initialize_optimizer_states
fused_optimizer.FP16_Optimizer.backward = Fp16OptimizerBackward
fused_optimizer.FP16_Optimizer.step = Fp16OptimizerStep
fused_optimizer.FP16_Optimizer.zero_grad = zero_grad
fused_optimizer.FP16_Optimizer.unscale_and_clip_grads = unscale_and_clip_grads
fused_optimizer.FP16_Optimizer.get_combine_weight_norm = get_combine_weight_norm
fused_optimizer.FP16_Optimizer.state_dict = state_dict
fused_optimizer.FP16_Optimizer.load_state_dict = load_state_dict
fused_optimizer.FP16_Optimizer.refresh_fp32_params = refresh_fp32_params
