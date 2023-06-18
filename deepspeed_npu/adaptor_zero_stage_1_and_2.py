import sys
import torch
import torch_npu
import deepspeed
from deepspeed import comm as dist

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import stage_1_and_2
from deepspeed.runtime.utils import is_model_parallel_parameter
from deepspeed.runtime.constants import PIPE_REPLICATED


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type),
        "torch.{}.FloatTensor".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


stage_1_and_2.split_half_float_double = split_half_float_double


def update_overflow_tracker_for_param_grad(self, param):
    if param.grad is not None and torch_npu._amp_foreach_non_finite_check_([param.grad.data]):
        self.local_overflow = True


def complete_grad_norm_calculation_for_cpu_offload(self, params):
    total_norm = 0.0
    norm_type = 2.0
    for p in params:
        # Pipeline parallelism may replicate parameters. Avoid multi-counting.
        if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
            continue

        if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
            param_id = self.get_param_id(p)
            # as some model have trainable parameters but skipped in training,
            # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
            # so they have no norm_for_param_grads
            if param_id in self.norm_for_param_grads:
                param_norm = self.norm_for_param_grads[param_id]
                total_norm += param_norm.item() ** 2
            else:
                # As unused parameters in modules may not be expected sometimes,
                # add an explicit error msg when it occurred and an option to
                # avoid the error
                assert self.ignore_unused_parameters, """
                    This assert indicates that your module has parameters that
                    were not used in producing loss.
                    You can avoid this assert by
                    (1) enable ignore_unused_parameters option in zero_optimization config;
                    (2) making sure all trainable parameters and `forward` function
                        outputs participate in calculating loss.
                """

    # Sum across all model parallel GPUs.
    total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
    dist.all_reduce(total_norm_npu, op=dist.ReduceOp.SUM, group=self.dp_process_group)

    self._model_parallel_all_reduce(tensor=total_norm_npu, op=dist.ReduceOp.SUM)

    total_norm = total_norm_npu[0].item() ** (1. / norm_type)

    overflow = torch_npu._amp_foreach_non_finite_check_([total_norm_npu])
    overflow_npu = get_accelerator().IntTensor([overflow])
    dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

    self._model_parallel_all_reduce(tensor=overflow_npu, op=dist.ReduceOp.MAX)

    if overflow_npu.item() or total_norm == float('inf') or \
            total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def get_grad_norm_direct(self, gradients, params, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        # Take max across all GPUs.
        self._model_parallel_all_reduce(tensor=total_norm_npu, op=dist.ReduceOp.MAX)
        total_norm = total_norm_npu[0].item()
    else:
        total_norm = 0.0
        # if dist.get_rank() == 0:
        #    logger.info(f"Total Norm beginning {total_norm}")
        for g, p in zip(gradients, params):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_norm = g.data.double().norm(2)
                total_norm += param_norm.item() ** 2
        # Sum across all model parallel GPUs.
        total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_npu, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_npu, op=dist.ReduceOp.SUM)

        total_norm = total_norm_npu[0].item() ** (1. / norm_type)

    overflow = torch_npu._amp_foreach_non_finite_check_([total_norm_npu])
    overflow_npu = get_accelerator().IntTensor([overflow])
    dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

    self._model_parallel_all_reduce(tensor=overflow_npu, op=dist.ReduceOp.MAX)

    if overflow_npu.item() or total_norm == float('inf') or \
            total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def has_overflow_serial(self, params, is_grad_list=False):
    grads = [p.grad.data for p in params if p.grad is not None]
    return torch_npu._amp_foreach_non_finite_check_(grads)


def has_overflow_partitioned_grads_serial(self):
    grads = []
    for i in range(len(self.bit16_groups)):
        for j, grad in enumerate(self.averaged_gradients[i]):
            if grad is not None:
                grads.append(grad.data)
    return torch_npu._amp_foreach_non_finite_check_(grads)


def has_overflow(self, partition_gradients=True):
    if partition_gradients:
        overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial(
        )
    else:
        params = []
        for group in self.bit16_groups:
            for param in group:
                params.append(param)

        overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)

    overflow_npu = get_accelerator().IntTensor([overflow])
    dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

    # Since each model parallel GPU carries only part of the model,
    # make sure overflow flag is synced across all the model parallel GPUs
    self._model_parallel_all_reduce(tensor=overflow_npu,
                                    op=dist.ReduceOp.MAX)

    overflow = overflow_npu[0].item()
    return bool(overflow)


deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.update_overflow_tracker_for_param_grad = update_overflow_tracker_for_param_grad
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.complete_grad_norm_calculation_for_cpu_offload = complete_grad_norm_calculation_for_cpu_offload
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.get_grad_norm_direct = get_grad_norm_direct
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow_serial = has_overflow_serial
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow_partitioned_grads_serial = has_overflow_partitioned_grads_serial
deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow = has_overflow