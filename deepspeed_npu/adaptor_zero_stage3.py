from typing import List
import torch
import torch_npu
import deepspeed
from torch import Tensor
from torch.nn import Parameter
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import is_model_parallel_parameter
from deepspeed.runtime.utils import see_memory_usage
from . import FLAG_SUPPORT_INF_NAN

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf


def complete_grad_norm_calculation_for_cpu_offload(self, params):
    total_norm = 0.0
    norm_type = 2.0
    for p in params:
        if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
            param_id = self.get_param_id(p)
            if param_id in self.norm_for_param_grads.keys():
                param_norm = self.norm_for_param_grads[param_id]
                total_norm += param_norm.item()**2

    # Sum across all model parallel GPUs.
    total_norm_npu = get_accelerator().FloatTensor([float(total_norm)])

    dist.all_reduce(total_norm_npu, op=dist.ReduceOp.SUM, group=self.dp_process_group)
    self._model_parallel_all_reduce(tensor=total_norm_npu, op=dist.ReduceOp.SUM)

    total_norm = total_norm_npu[0].item()**(1. / norm_type)

    overflow_npu = False
    if not FLAG_SUPPORT_INF_NAN:
        overflow = torch_npu._amp_foreach_non_finite_check_([total_norm_npu])
        overflow_npu = torch.npu.IntTensor([overflow])
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=overflow_npu, op=dist.ReduceOp.MAX)
        overflow_npu = overflow_npu.item()

    if overflow_npu or total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
    offload_fp32_gradients = {}
    offload_fp32_offsets = {}
    buffers = []
    for param, grad_partition in zip(params_to_release, grad_partitions):

        contains_real_data = param.partition_numel() * dist.get_rank(self.dp_process_group) < param.ds_numel
        if not contains_real_data:
            # this grad partition is empty - don't need to do anything
            param.grad = None
            continue

        # move or accumulate gradient partition to target buffer
        grad_buffer = self._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
        buffers.append(grad_buffer)
        if self.micro_step_id == 0:  # don't accumulate
            grad_buffer.copy_(grad_partition, non_blocking=True)
            # ensure grad buffer is a CUDA buffer to speed up the next few
            # operations and so it can be used asynchronously
            grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
        elif get_accelerator().on_accelerator(grad_buffer):
            grad_buffer.add_(grad_partition)
        else:
            # if dst is CPU, copy first to src device, do the addition
            # there, then move back to dst. adding directly to cpu is very slow
            cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            cuda_grad_buffer.add_(grad_partition)
            grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
            # ensure grad buffer is a CUDA buffer to speed up the next few
            # operations and so it can be used asynchronously
            grad_buffer = cuda_grad_buffer

        if not FLAG_SUPPORT_INF_NAN:
            self.inf_or_nan_tracker[0] = self.inf_or_nan_tracker[0] or torch_npu.npu.get_npu_overflow_flag()
        else:
            if hasattr(self.inf_or_nan_tracker, "logical_or_"):
                self.inf_or_nan_tracker.logical_or_(torch.isinf(grad_buffer).any())
                self.inf_or_nan_tracker.logical_or_(torch.isnan(grad_buffer).any())
            else:
                # logical_or_ not available in older versions of pytorch
                self.inf_or_nan_tracker += torch.isinf(grad_buffer).any()
                self.inf_or_nan_tracker += torch.isnan(grad_buffer).any()
                self.inf_or_nan_tracker = self.inf_or_nan_tracker > 0

        # offload the gradient partition if applicable
        if self.offload_optimizer:
            i, dest_offset, _ = self.grad_position[self.get_param_id(param)]

            if self.is_gradient_accumulation_boundary:
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                if self._swappable_optimizer_subgroup(i):
                    if not i in offload_fp32_gradients.keys():
                        offload_fp32_gradients[i] = []
                        offload_fp32_offsets[i] = []

                    offload_fp32_gradients[i].append(grad_buffer.float())
                    offload_fp32_offsets[i].append(dest_offset)
                else:
                    fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                        0, dest_offset, grad_buffer.numel())
                    fp32_grad_tensor.copy_(grad_buffer)

        # free the gradient
        param.grad.record_stream(get_accelerator().current_stream())
        param.grad = None

    if self.offload_optimizer and self.swap_optimizer:
        for i in offload_fp32_gradients.keys():
            self.optimizer_swapper.swap_out_gradients(parameter=self.fp32_partitioned_groups_flat[i],
                                                      gradient_offsets=offload_fp32_offsets[i],
                                                      gradient_tensors=offload_fp32_gradients[i])
    return buffers


def get_grad_norm_direct(self, gradients, params, norm_type=2):
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        # Take max across all GPUs.
        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
        total_norm = total_norm_cuda[0].item()
    else:
        # if dist.get_rank() == 0:
        #    logger.info(f"Total Norm beginning {total_norm}")
        grad_norms = []
        for g, p in zip(gradients, params):
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                grad_norms.append(g.to(get_accelerator().device_name(), non_blocking=True).double().norm(2))

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.sum(torch.pow(torch.stack(grad_norms), 2))

        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

        total_norm = total_norm_cuda.item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def has_overflow_serial(self, params, is_grad_list=False):
    if not FLAG_SUPPORT_INF_NAN:
        grads = [p.grad.data for p in params if p.grad is not None]
        return torch_npu._amp_foreach_non_finite_check_(grads)

    for p in params:
        if p.grad is not None and self._has_inf_or_nan(p.grad.data):
            return True

    return False


def has_overflow_partitioned_grads_serial(self):
    if not FLAG_SUPPORT_INF_NAN:
        grads = []
        for i in range(len(self.fp16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None:
                    grads.append(grad.data)
        return torch_npu._amp_foreach_non_finite_check_(grads)

    for i in range(len(self.fp16_groups)):
        for j, grad in enumerate(self.averaged_gradients[i]):
            if grad is not None and self._has_inf_or_nan(grad.data, j):
                return True
    return False


def has_overflow(self, partition_gradients=True):
    if partition_gradients:
        with get_accelerator().stream(self.reduce_and_partition_stream):
            self.local_overflow = bool(self.inf_or_nan_tracker.item())
            self.inf_or_nan_tracker.zero_()

        overflow = self.local_overflow
        #overflow = self.has_overflow_partitioned_grads_serial()
        overflow_npu = get_accelerator().IntTensor([overflow])
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

    else:
        params = []
        for group in self.fp16_groups:
            for param in group:
                params.append(param)

        overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
        overflow_npu = get_accelerator().IntTensor([overflow])

    # Since each model parallel GPU carries only part of the model,
    # make sure overflow flag is synced across all the model parallel GPUs
    if not FLAG_SUPPORT_INF_NAN:
        dist.all_reduce(overflow_npu, op=dist.ReduceOp.MAX, group=self.dp_process_group)
    self._model_parallel_all_reduce(tensor=overflow_npu, op=dist.ReduceOp.MAX)

    overflow = overflow_npu[0].item()
    return bool(overflow)


def backward(self, loss, retain_graph=False):
    if not FLAG_SUPPORT_INF_NAN:
        torch_npu.npu.clear_npu_overflow_flag()
    if self.swap_optimizer:
        self.optimizer_swapper.pre_backward()

    see_memory_usage(f"Before backward", force=False)

    if self.custom_loss_scaler:
        scaled_loss = self.external_loss_scale * loss
        scaled_loss.backward()
    else:
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    self._get_param_coordinator(training=True).reset_step()

    if self.swap_optimizer:
        self.optimizer_swapper.post_backward()


deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.complete_grad_norm_calculation_for_cpu_offload = complete_grad_norm_calculation_for_cpu_offload
deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.partition_grads = partition_grads
deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow_serial = has_overflow_serial
deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow_partitioned_grads_serial = has_overflow_partitioned_grads_serial
deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow = has_overflow
