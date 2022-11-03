import sys
import torch
import torch_npu
from torch._six import inf
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_global_rank
from deepspeed.runtime.zero import stage_1_and_2
from deepspeed.runtime.utils import is_model_parallel_parameter
from deepspeed.moe.utils import is_moe_param


def split_half_float_double(tensors):
    dtypes = [
        "torch.npu.HalfTensor",
        "torch.npu.FloatTensor"
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets

stage_1_and_2.split_half_float_double = split_half_float_double

class DeepSpeedZeroOptimizerNpu(stage_1_and_2.DeepSpeedZeroOptimizer):
    def __init__(self,
                 init_optimizer,
                 timers,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 cpu_offload=False,
                 mpu=None,
                 clip_grad=0.0,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):
        super().__init__(
                 init_optimizer,
                 timers,
                 static_loss_scale,
                 dynamic_loss_scale,
                 dynamic_loss_args,
                 verbose,
                 contiguous_gradients,
                 reduce_bucket_size,
                 allgather_bucket_size,
                 dp_process_group,
                 expert_parallel_group,
                 expert_data_parallel_group,
                 reduce_scatter,
                 overlap_comm,
                 cpu_offload,
                 mpu,
                 clip_grad,
                 communication_data_type,
                 postscale_gradients,
                 gradient_predivide_factor,
                 gradient_accumulation_steps,
                 ignore_unused_parameters,
                 partition_grads,
                 round_robin_gradients,
                 has_moe_layers,
                 fp16_master_weights_and_gradients,
                 elastic_checkpoint)
        if self.cpu_offload:
            self.grads_need_check_overflow = []
    
    def average_tensor(self, tensor):
        if self.overlap_comm:
            torch.npu.synchronize()
            stream = self.reduction_stream
        else:
            stream = torch.npu.current_stream()

        with torch.npu.stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id = -1

            process_group = self.dp_process_group
            # count = 0
            for i, param, param_id in self.params_in_ipg_bucket:

                process_group = self.dp_process_group
                #Averages gradients at parameter level if ipg has a moe param
                #Otherwise averaging is done at the entire buffer level at the end of the loop
                # MoE param have different groups
                if self.ipg_bucket_has_moe_params:
                    process_group = self.expert_dp_process_group[
                        param.group_name] if is_moe_param(
                            param) else self.dp_process_group
                    param.grad.data.div_(dist.get_world_size(group=process_group))

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]
                # Get all partition ids + their offsets
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # if dist.get_rank() == 0 and count < 100:
                    #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                    # count += 1

                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    if partition_id == prev_id:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id = partition_id

            if not self.ipg_bucket_has_moe_params:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))

            async_handles = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                # if dist.get_rank() == 0:
                #     print(f"Rank {dist.get_rank()} rank offset id {i} real dp size {dist.get_world_size(group=real_dp_process_group[i])} and dst: {dst}")
                # dist.barrier()
                #dist.barrier()

                dst_rank = _get_global_rank(real_dp_process_group[i], dst)
                # async_handle = dist.reduce(grad_slice,
                #                            dst=dst_rank,
                #                            group=real_dp_process_group[i],
                #                            async_op=True)
                # ASCEND AVOID
                tmp = grad_slice.clone()
                async_handle = dist.reduce(tmp,
                                           dst=dst_rank,
                                           group=real_dp_process_group[i],
                                           async_op=False)
                grad_slice.data.copy_(tmp)
                #async_handles.append(async_handle)

            #for handle in async_handles:
                #handle.wait()

    def update_overflow_tracker_for_param_grad(self, param):
        if param.grad is not None:
            self.grads_need_check_overflow.append(param.grad.data)

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
                # so they have no norm_for_param_grads
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item()**2
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

    def reduce_ipg_grads(self):
        if self.contiguous_gradients:
            if self.extra_large_param_to_reduce is not None:
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                _, _, param_id = self.params_in_ipg_bucket[0]
                assert self.get_param_id(
                    self.extra_large_param_to_reduce) == param_id, "param in ipg bucket does not match extra-large param"
                self.average_tensor(self.extra_large_param_to_reduce.grad.view(-1))
                self.extra_large_param_to_reduce = None
            else:
                self.average_tensor(self.ipg_buffer[self.ipg_index])
        else:
            self.buffered_reduce_fallback(
                None,
                self.grads_in_ipg_bucket,
                elements_per_buffer=self.elements_in_ipg_bucket)

        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            torch.npu.synchronize()
            #            stream = self.copy_grad_stream
            stream = torch.npu.current_stream()
        else:
            stream = torch.npu.current_stream()

        with torch.npu.stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:

                assert self.params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"

                self.params_already_reduced[param_id] = True

                if self.partition_gradients:
                    if not self.is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.previous_reduced_grads is None:
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            param.grad = None  #only if self.partition_gradients
                    elif self.contiguous_gradients:
                        self.copy_grads_in_partition(param)
                else:  # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                        self.copy_grads_in_partition(param)

            if self.cpu_offload:
                self.local_overflow = torch._amp_foreach_non_finite_check_(self.grads_need_check_overflow)
                self.grads_need_check_overflow = []

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0

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
            total_norm = 0.0
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
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

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False
        self.grads_need_check_overflow = []

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params, is_grad_list=False):
        grads = [p.grad.data for p in params if p.grad is not None]
        return torch._amp_foreach_non_finite_check_(grads)

    def has_overflow_partitioned_grads_serial(self):
        grads = []
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None:
                    grads.append(grad.data)
        return torch._amp_foreach_non_finite_check_(grads)

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
    if 'deepspeed' in k and hasattr(v, 'DeepSpeedZeroOptimizer'):
        setattr(v, 'DeepSpeedZeroOptimizer', DeepSpeedZeroOptimizerNpu)