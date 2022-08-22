from deepspeed.runtime import engine
from deepspeed.utils import logger
import hashlib
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
                    '''
                    # ASCEND VOID OPT
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    optimizer = DeepSpeedCPUAdam(model_parameters,
                                                    **optimizer_parameters,
                                                    adamw_mode=effective_adam_w_mode)
                    '''
                    optimizer = torch.optim.Adam(
                        model_parameters,
                        **optimizer_parameters,
                    )
            else:
                # ASCEND VOID OPT
                # from deepspeed.ops.adam import FusedAdam

                optimizer = torch.optim.Adam(
                    model_parameters,
                    **optimizer_parameters,
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

engine._configure_basic_optimizer = _configure_basic_optimizer
engine._checkpoint_tag_validation = _checkpoint_tag_validation
