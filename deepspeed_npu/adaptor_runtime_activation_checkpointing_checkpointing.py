import torch
import torch_npu
import deepspeed.runtime.activation_checkpointing.checkpointing

from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
from deepspeed.runtime.activation_checkpointing.checkpointing import gather_partitioned_activations, detach_variable, \
    merge_tensors, get_cuda_rng_tracker, is_activation_to_checkpoint, extract_tensors
from deepspeed.runtime.utils import copy_to_device, move_to_device, see_memory_usage, bwc_tensor_model_parallel_rank
from . import FLAG_SUPPORT_INF_NAN

CKPT_INIT_FLAG = False
CKPT_OVERFLOW_FLAG = False
CKPT_CONST_VAR = None


def backward(ctx, *grads):
    with torch.autograd.profiler.record_function('checkpoint_backward'):
        global timers
        see_memory_usage("In backward", force=False)
        # removing pointers to the contiguous buffer memory
        # so that they can be garbage collected once the checkpoints
        # have been used
        if deepspeed.runtime.activation_checkpointing.checkpointing.SYNCHRONIZE:
            torch.cuda.synchronize()
        if deepspeed.runtime.activation_checkpointing.checkpointing.PROFILE_TIME:
            timers('backward').start()

        if deepspeed.runtime.activation_checkpointing.checkpointing.CONTIGUOUS_CHECKPOINTING:
            for buffers in deepspeed.runtime.activation_checkpointing.checkpointing.contiguous_data_buffers:
                buffers = []

            # frees up all the pointers to the checkpoints except for the ones
            # stored by save for backward
            deepspeed.runtime.activation_checkpointing.checkpointing.contiguous_data_buffers = []
            deepspeed.runtime.activation_checkpointing.checkpointing.contiguous_size_buffers = []
            deepspeed.runtime.activation_checkpointing.checkpointing.data_offsets = []
            deepspeed.runtime.activation_checkpointing.checkpointing.size_offsets = []

        see_memory_usage("In backward checkpointing code", force=False)
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")

        if deepspeed.runtime.activation_checkpointing.checkpointing.PARTITION_ACTIVATIONS:
            # with torch.cuda.stream(transport_stream):
            inputs = gather_partitioned_activations(
                ctx.deepspeed_saved_tensors,
                device=deepspeed.runtime.activation_checkpointing.checkpointing.cuda_device \
                    if deepspeed.runtime.activation_checkpointing.checkpointing.CPU_CHECKPOINT else None)
            detached_inputs = detach_variable(inputs)
        elif deepspeed.runtime.activation_checkpointing.checkpointing.CPU_CHECKPOINT:
            inputs = move_to_device(ctx.deepspeed_saved_tensors,
                                    deepspeed.runtime.activation_checkpointing.checkpointing.cuda_device,
                                    is_activation_to_checkpoint)
            detached_inputs = detach_variable(inputs)
        else:
            inputs = ctx.deepspeed_saved_tensors
            detached_inputs = detach_variable(inputs)

        # Add non tensor input args
        detached_inputs = merge_tensors(tensor_objects=detached_inputs,
                                        non_tensor_objects=ctx.non_tensor_args,
                                        tensor_flags=ctx.tensor_flags)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # if PARTITION_ACTIVATIONS:
        #     current_stream=torch.cuda.current_stream()
        #     current_stream.wait_stream(transport_stream)

        see_memory_usage("In backward checkpointing code before forward", force=False)

        if not FLAG_SUPPORT_INF_NAN:
            global CKPT_INIT_FLAG, CKPT_OVERFLOW_FLAG, CKPT_CONST_VAR
            if not CKPT_INIT_FLAG:
                CKPT_INIT_FLAG = True
                CKPT_CONST_VAR = torch.tensor([65504.], dtype=torch.float16).npu()

            CKPT_OVERFLOW_FLAG = torch_npu.npu.get_npu_overflow_flag()
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)
                torch.npu.clear_npu_overflow_flag()
        else:
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        see_memory_usage("In backward checkpointing code after forward", force=False)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Filter out non tensor outputs
        outputs, _, _ = extract_tensors(all_objects=outputs)

        # Construct arguments to autograd.backward().
        # This is usually just outputs and grads, but forward() can return tensors that
        # are not differentiable.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)

        see_memory_usage("In backward checkpointing code before backward", force=False)

        torch.autograd.backward(output_tensors, grad_tensors)

        # Force clear our stashed tensors to prevent a memory leak in certain scenarios
        ctx.deepspeed_saved_tensors = None
        ctx.non_tensor_args = None
        ctx.tensor_flags = None

        see_memory_usage("After backward checkpointing code after backward", force=False)

        if deepspeed.runtime.activation_checkpointing.checkpointing.PROFILE_TIME:
            timers('backward').stop()
            timers.log(['backward'])
        if deepspeed.runtime.activation_checkpointing.checkpointing.SYNCHRONIZE:
            torch.cuda.synchronize()
        ret_list = [None, None]  # first None for ctx
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        if not FLAG_SUPPORT_INF_NAN:
            temp = torch_npu.npu.get_npu_overflow_flag()
            CKPT_OVERFLOW_FLAG = CKPT_OVERFLOW_FLAG or temp
            CKPT_CONST_VAR + CKPT_OVERFLOW_FLAG * 10000

    return tuple(ret_list)


def _set_cuda_rng_state(new_state, device=-1):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]  # not replace with torch_npu?
            default_generator.set_state(new_state)

    _lazy_call(cb)


def _configure_defaults():
    deepspeed.runtime.activation_checkpointing.checkpointing.PARTITION_ACTIVATIONS = False
    deepspeed.runtime.activation_checkpointing.checkpointing.CONTIGUOUS_CHECKPOINTING = False
    deepspeed.runtime.activation_checkpointing.checkpointing.num_layers = False
    deepspeed.runtime.activation_checkpointing.checkpointing.CPU_CHECKPOINT = False
    deepspeed.runtime.activation_checkpointing.checkpointing.SYNCHRONIZE = False
    deepspeed.runtime.activation_checkpointing.checkpointing.PROFILE_TIME = False
    deepspeed.runtime.activation_checkpointing.checkpointing.deepspeed_checkpointing_enabled = True


deepspeed.runtime.activation_checkpointing.checkpointing.CheckpointFunction.backward = backward
deepspeed.runtime.activation_checkpointing.checkpointing._set_cuda_rng_state = _set_cuda_rng_state
deepspeed.runtime.activation_checkpointing.checkpointing._configure_defaults = _configure_defaults
