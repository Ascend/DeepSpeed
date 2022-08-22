import torch
import torch_npu
from deepspeed.runtime.zero import stage_1_and_2
from torch.distributed.distributed_c10d import _get_global_rank
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
            async_handle = dist.reduce(grad_slice,
                                        dst=dst_rank,
                                        group=real_dp_process_group[i],
                                        async_op=False)
            #async_handles.append(async_handle)

        #for handle in async_handles:
            #handle.wait()

stage_1_and_2.average_tensor = average_tensor