import hashlib
import torch
import torch.distributed as dist
from deepspeed.runtime import engine
from deepspeed.utils import logger, log_dist
from deepspeed_npu.adaptor_ops_adam_fused_adam import FusedAdamNPU
from deepspeed.runtime.config import DeepSpeedConfig, DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer


try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False
    pass


def split_half_float_double_sparse(tensors):
    supported_types = [
        "torch.npu.HalfTensor",
        "torch.npu.FloatTensor"
    ]

    for t in tensors:
        assert t.type() in supported_types, f"attempting to reduce an unsupported grad type: {t.type()}"

    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets

engine.split_half_float_double_sparse = split_half_float_double_sparse

def _configure_basic_optimizer(self, model_parameters):
    optimizer_parameters = self.optimizer_params()
    if optimizer_parameters is None:
        optimizer_parameters = {}
    # print(optimizer_parameters.keys())
    if "max_grad_norm" in optimizer_parameters.keys():
        raise ValueError(
            "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
        )

    if self.optimizer_name() in [ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
        torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
        adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

        # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
        effective_adam_w_mode = self.optimizer_name(
        ) == ADAMW_OPTIMIZER or adam_w_mode

        if torch_adam:
            if not effective_adam_w_mode:
                optimizer = torch.optim.Adam(model_parameters,
                                                **optimizer_parameters)
            else:
                optimizer = torch.optim.AdamW(model_parameters,
                                                **optimizer_parameters)
        else:
            if self.zero_cpu_offload():
                if self.optimizer_name() == ADAGRAD_OPTIMIZER:
                    '''
                    # ASCEND VOID OPT
                    from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                    optimizer = DeepSpeedCPUAdagrad(model_parameters,
                                                    **optimizer_parameters)
                    '''
                    optimizer = torch.optim.Adagrad(
                        model_parameters,
                        **optimizer_parameters,
                    )
                else:
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    optimizer = DeepSpeedCPUAdam(model_parameters,
                                                    **optimizer_parameters,
                                                    adamw_mode=effective_adam_w_mode)
            else:
                optimizer = FusedAdamNPU(
                    model_parameters,
                    **optimizer_parameters,
                    adam_w_mode=effective_adam_w_mode
                )

    elif self.optimizer_name() == LAMB_OPTIMIZER:
        '''
        # ASCEND VOID OPT
        from deepspeed.ops.lamb import FusedLamb
        optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        '''
        from apex.optimizers import Lamb

        optimizer = Lamb(model_parameters, **optimizer_parameters)
    elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
        assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
        from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

        optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
        if not self.fp16_enabled():
            logger.warning(
                f"Currently the convergence of 1-bit Adam is only verified under FP16"
            )
    elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
        assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
        from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

        optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
        if not self.fp16_enabled():
            logger.warning(
                f"Currently the convergence of 1-bit Lamb is only verified under FP16"
            )
    else:
        torch_optimizer = getattr(torch.optim, self.optimizer_name())
        optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
    return optimizer

def _checkpoint_tag_validation(self, tag):
    if self.checkpoint_tag_validation_enabled():
        s_hash = hashlib.sha1(tag.encode())
        bhash = torch.ByteTensor([s_hash.digest()]).flatten().to(self.device)
        max_bhash = bhash.clone()
        min_bhash = bhash.clone()

        # ASCEND AVOID
        max_bhash = max_bhash.int()
        min_bhash = min_bhash.int()
        dist.all_reduce(max_bhash, op=torch.distributed.ReduceOp.MAX)
        dist.all_reduce(min_bhash, op=torch.distributed.ReduceOp.MIN)

        max_bhash = max_bhash.byte()
        min_bhash = min_bhash.byte()

        valid = all(min_bhash == bhash) and all(max_bhash == bhash)
        msg = (
            f"[rank={dist.get_rank()}] The checkpoint tag name '{tag}' is not consistent across "
            "all ranks. Including rank unique information in checkpoint tag could cause issues when "
            "restoring with different world sizes.")
        if self.checkpoint_tag_validation_fail():
            assert valid, msg
        elif not valid:
            logger.warning(msg)


def _configure_fp16_optimizer(self, optimizer):
    initial_dynamic_scale = self.initial_dynamic_scale()
    dynamic_loss_args = self.dynamic_loss_scale_args()
    clip_grad = self.gradient_clipping()
    if APEX_INSTALLED:
        fused_opts = (apex.optimizers.FusedAdam, FusedAdamNPU)
    else:
        fused_opts = FusedAdamNPU

    if isinstance(optimizer, fused_opts) \
            or self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
        if self.dynamic_loss_scale():
            log_dist("Creating fp16 optimizer with dynamic loss scale", ranks=[0])
            timers = self.timers if self.wall_clock_breakdown() else None
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                dynamic_loss_scale=True,
                initial_dynamic_scale=initial_dynamic_scale,
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                timers=timers,
            )
        else:
            log_dist(
                "Creating fp16 optimizer with static loss scale: {}".format(
                    self.loss_scale()),
                ranks=[0],
            )
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=self.loss_scale(),
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
            )
    else:
        log_dist("Creating fp16 unfused optimizer with dynamic loss scale",
                 ranks=[0])
        optimizer = FP16_UnfusedOptimizer(
            optimizer,
            deepspeed=self,
            static_loss_scale=self.loss_scale(),
            dynamic_loss_scale=self.dynamic_loss_scale(),
            dynamic_loss_args=dynamic_loss_args,
            mpu=self.mpu,
            clip_grad=clip_grad,
            fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER,
        )

    return optimizer


engine.DeepSpeedEngine._configure_basic_optimizer = _configure_basic_optimizer
engine.DeepSpeedEngine._checkpoint_tag_validation = _checkpoint_tag_validation
engine.DeepSpeedEngine._configure_fp16_optimizer = _configure_fp16_optimizer
