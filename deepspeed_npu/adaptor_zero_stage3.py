from typing import Deque, Dict, Iterable, Set, Tuple, List
import sys
import torch
from torch import Tensor
from math import inf
from torch.nn import Parameter
import torch.distributed as dist
from deepspeed.runtime.utils import is_model_parallel_parameter
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3


class DeepSpeedZeroOptimizer_Stage3Npu(DeepSpeedZeroOptimizer_Stage3):
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
        total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
        torch.distributed.all_reduce(total_norm_npu,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_npu,
                                        op=torch.distributed.ReduceOp.SUM)

        total_norm = total_norm_npu[0].item()**(1. / norm_type)

        overflow = torch._amp_foreach_non_finite_check_([total_norm_npu])
        overflow_npu = torch.npu.IntTensor([overflow])
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX)

        if overflow_npu.item() or total_norm == float('inf') or \
            total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    def _DeepSpeedZeroOptimizer_Stage3__partition_grads(self,
                          params_to_release: List[Parameter],
                          grad_partitions: List[Tensor]) -> None:
        grad_buffer_lst = []
        for param, grad_partition in zip(params_to_release, grad_partitions):
            if param.ds_tensor.ds_numel * dist.get_rank(
                    self.dp_process_group) > param.ds_numel:
                # this grad partition is empty - don't need to do anything
                continue

            # move or accumulate gradient partition to target buffer
            grad_buffer = self._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id].narrow(
                0,
                0,
                grad_partition.numel())
            if self.micro_step_id == 0:  # don't accumulate
                grad_buffer.copy_(grad_partition, non_blocking=True)
                # ensure grad buffer is a NPU buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            elif grad_buffer.is_npu:
                grad_buffer.add_(grad_partition)
            else:
                # if dst is CPU, copy first to src device, do the addition
                # there, then move back to dst. adding directly to cpu is very slow
                npu_grad_buffer = grad_buffer.to(grad_partition.device,
                                                  non_blocking=True)
                npu_grad_buffer.add_(grad_partition)
                grad_buffer.copy_(npu_grad_buffer, non_blocking=True)
                # ensure grad buffer is a NPU buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = npu_grad_buffer

            grad_buffer_lst.append(grad_buffer.data)

            '''
            # ASCEND AVOID OVERFLOW
            if hasattr(self.__inf_or_nan_tracker, "logical_or_"):
                self.__inf_or_nan_tracker.logical_or_(torch.isinf(grad_buffer).any())
                self.__inf_or_nan_tracker.logical_or_(torch.isnan(grad_buffer).any())
            else:
                # logical_or_ not available in older versions of pytorch
                self.__inf_or_nan_tracker += torch.isinf(grad_buffer).any()
                self.__inf_or_nan_tracker += torch.isnan(grad_buffer).any()
                self.__inf_or_nan_tracker = self.__inf_or_nan_tracker > 0
            '''

            # offload the gradient partition if applicable
            if self.offload_optimizer:
                i, dest_offset, _ = self.grad_position[self.get_param_id(param)]
                offload_fp32_gradients = {}
                offload_fp32_offsets = {}

                if self.is_gradient_accumulation_boundary:
                    self.norm_for_param_grads[self.get_param_id(
                        param)] = self._constant_buffered_norm2(grad_buffer)

                    if self._swappable_optimizer_subgroup(i):
                        if not i in offload_fp32_gradients.keys():
                            offload_fp32_gradients[i] = []
                            offload_fp32_offsets[i] = []

                        offload_fp32_gradients[i].append(grad_buffer.float())
                        offload_fp32_offsets[i].append(dest_offset)
                    else:
                        fp32_grad_tensor = self.fp32_partitioned_groups_flat[
                            i].grad.narrow(0,
                                           dest_offset,
                                           grad_buffer.numel())
                        fp32_grad_tensor.copy_(grad_buffer)

            # free the gradient
            param.grad.record_stream(torch.npu.current_stream())
            param.grad = None

        overflow_npu = torch._amp_foreach_non_finite_check_(grad_buffer_lst)
        self._DeepSpeedZeroOptimizer_Stage3__inf_or_nan_tracker = torch.BoolTensor([overflow_npu]).to(torch.npu.current_device())

        if self.offload_optimizer and self.swap_optimizer:
            for i in offload_fp32_gradients.keys():
                self.optimizer_swapper.swap_out_gradients(
                    parameter=self.fp32_partitioned_groups_flat[i],
                    gradient_offsets=offload_fp32_offsets[i],
                    gradient_tensors=offload_fp32_gradients[i])

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
            total_norm_npu = torch.npu.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_npu,
                                            op=torch.distributed.ReduceOp.MAX)
            total_norm = total_norm_npu[0].item()
        else:
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            grad_norms = []
            for g, p in zip(gradients, params):
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    grad_norms.append(g.npu(non_blocking=True).double().norm(2))

            # Sum across all model parallel GPUs.
            total_norm_npu = torch.sum(torch.pow(torch.stack(grad_norms), 2))
            torch.distributed.all_reduce(total_norm_npu,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_npu,
                                            op=torch.distributed.ReduceOp.SUM)

            total_norm = total_norm_npu.item()**(1. / norm_type)
        
        overflow = torch._amp_foreach_non_finite_check_([total_norm_npu])
        overflow_npu = torch.npu.IntTensor([overflow])
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX)

        if overflow_npu.item() or total_norm == float('inf') or \
            total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    def has_overflow_serial(self, params, is_grad_list=False):
        grads = [p.grad.data for p in params if p.grad is not None]
        return torch._amp_foreach_non_finite_check_(grads)

    def has_overflow_partitioned_grads_serial(self):
        grads = []
        for i in range(len(self.fp16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None:
                    grads.append(grad.data)
        return torch._amp_foreach_non_finite_check_(grads)

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            with torch.npu.stream(self._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_stream):
                self.local_overflow = bool(self._DeepSpeedZeroOptimizer_Stage3__inf_or_nan_tracker.item())
                self._DeepSpeedZeroOptimizer_Stage3__inf_or_nan_tracker.zero_()

            overflow = self.local_overflow
            #overflow = self.has_overflow_partitioned_grads_serial()
        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)

        overflow_npu = torch.npu.IntTensor([overflow])
        torch.distributed.all_reduce(overflow_npu,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.dp_process_group)

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_npu,
                                        op=torch.distributed.ReduceOp.MAX)

        overflow = overflow_npu[0].item()
        return bool(overflow)

for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'DeepSpeedZeroOptimizer_Stage3'):
        setattr(v, 'DeepSpeedZeroOptimizer_Stage3', DeepSpeedZeroOptimizer_Stage3Npu)
