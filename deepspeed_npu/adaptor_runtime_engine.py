import torch
import deepspeed
from deepspeed.utils import logger
from deepspeed.runtime.config import (ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER,
    ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT,
    ZERO_ONE_ADAM_OPTIMIZER)


def _configure_basic_optimizer(self, model_parameters):
    optimizer_parameters = self.optimizer_params()
    if optimizer_parameters is None:
        optimizer_parameters = {}
    # print(optimizer_parameters.keys())
    if "max_grad_norm" in optimizer_parameters.keys():
        raise ValueError(
            "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping'"
        )

    if self.optimizer_name() in [ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
        torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
        adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

        # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
        effective_adam_w_mode = self.optimizer_name() == ADAMW_OPTIMIZER or adam_w_mode

        if torch_adam:
            if not effective_adam_w_mode:
                optimizer = torch.optim.Adam(model_parameters, **optimizer_parameters)
            else:
                optimizer = torch.optim.AdamW(model_parameters, **optimizer_parameters)
        else:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                optimizer = DeepSpeedCPUAdam(model_parameters, **optimizer_parameters, adamw_mode=effective_adam_w_mode)
            else:
                # ASCEND VOID OPT
                from deepspeed.ops.adam import FusedAdam
                optimizer = FusedAdam(
                    model_parameters,
                    **optimizer_parameters,
                    adam_w_mode=effective_adam_w_mode,
                )
    elif self.optimizer_name() == ADAGRAD_OPTIMIZER:
        if self.zero_use_cpu_optimizer():
            # ASCEND VOID OPT
            # from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
            # optimizer = DeepSpeedCPUAdagrad(model_parameters, **optimizer_parameters)
            optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)
        else:
            optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)
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
            logger.warning(f"Currently the convergence of 1-bit Adam is only verified under FP16")
    elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
        assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
        from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

        optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
        if not self.fp16_enabled():
            logger.warning(f'Currently the convergence of 0/1 Adam is only verified under FP16')
    elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
        assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
        from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

        optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
        if not self.fp16_enabled():
            logger.warning(f"Currently the convergence of 1-bit Lamb is only verified under FP16")
    else:
        torch_optimizer = getattr(torch.optim, self.optimizer_name())
        optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
    return optimizer


deepspeed.runtime.engine.DeepSpeedEngine._configure_basic_optimizer = _configure_basic_optimizer
