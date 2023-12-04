import sys
import torch
import torch_npu
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.pipe.engine import _tensor_bytes
from deepspeed.runtime.pipe import p2p, engine, schedule
from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.utils import PartitionedTensor
from . import FLAG_SUPPORT_INF_NAN

ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool
]

DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}


def _exec_backward_pass(self, buffer_id):
    if not FLAG_SUPPORT_INF_NAN:
        clear_npu_overflow_flag()

    assert self.optimizer is not None, "must provide optimizer during " \
                                       "init in order to use backward"

    self.mem_status('BEFORE BWD', reset_max=True)

    # The last stage just runs backward on the loss using DeepSpeed's typical
    # mechanisms.
    if self.is_last_stage():
        super(engine.PipelineEngine, self).backward(self.loss)
        self.mem_status('AFTER BWD')
        return

    outputs = self.pipe_buffers['outputs'][buffer_id]

    if self.wall_clock_breakdown():
        self.timers('backward_microstep').start()
        self.timers('backward').start()
        self.timers('backward_inner_microstep').start()
        self.timers('backward_inner').start()

    # Reconstruct if we previously partitioned the output. We must be
    # careful to also restore the computational graph of the tensors we partitioned.
    if self.is_pipe_partitioned:
        if self.is_grad_partitioned:
            part_output = PartitionedTensor.from_meta(
                meta=outputs[0],
                local_part=outputs[1],
                group=self.grid.get_slice_parallel_group())
            self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
        else:
            # Already restored from partition
            self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
            outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

    grad_tensors = self.grad_layer
    if self.is_grad_partitioned:
        # print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
        part_grad = PartitionedTensor.from_meta(
            meta=self.grad_layer[0],
            local_part=self.grad_layer[1],
            group=self.grid.get_slice_parallel_group())
        grad_tensors = (part_grad.full(), *grad_tensors[2:])
        part_grad = None
        # print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

    if self.bfloat16_enabled() and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.clear_lp_grads()

    # This handles either a single tensor or tuple of tensors.
    if isinstance(outputs, tuple):
        out_tensors = [t for t in outputs if t.is_floating_point()]
        assert len(out_tensors) == len(grad_tensors)
        torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
    else:
        torch.autograd.backward(tensors=(outputs,), grad_tensors=(grad_tensors,))

    if self.bfloat16_enabled() and not self.is_last_stage():
        # manually call because we don't call optimizer.backward()
        self.optimizer.update_hp_grads(clear_lp_grads=False)

    # Free up the memory from the output of forward()
    self.pipe_buffers['output_tensors'][buffer_id] = None
    self.pipe_buffers['outputs'][buffer_id] = None
    grad_tensors = None

    if self.wall_clock_breakdown():
        self.timers('backward_inner').stop()
        self.timers('backward_inner_microstep').stop()
        self.timers('backward').stop()
        self.timers('backward_microstep').stop()

    self.mem_status('AFTER BWD')


def _exec_send_grads(self, buffer_id):
    if self.wall_clock_breakdown():
        self.timers('pipe_send_grad').start()

    inputs = self.pipe_buffers['inputs'][buffer_id]

    # Partition the gradient
    if self.is_grad_partitioned:
        if isinstance(inputs, tuple):
            first_input = inputs[0]
            assert all([torch.is_tensor(elt) for elt in inputs[1:]])
            # inputs_grad_tail = [elt.grad for elt in inputs[1:] if elt.grad is not None]
            inputs_grad_tail = [elt.grad for elt in inputs[1:]]
        elif torch.is_tensor(inputs):
            first_input = inputs
            inputs_grad_tail = []
        else:
            raise ValueError("expecting a tensor or a tuple of tensors")
        assert torch.is_tensor(first_input)
        part = PartitionedTensor(tensor=first_input.grad, group=self.grid.get_slice_parallel_group())

        inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

    # XXX Terrible hack
    # Drop the attention mask from the input buffer here. It does not have
    # a grad that needs to be communicated. We free the buffer immediately
    # after, so no need to restore it. The receiver also has a hack that skips
    # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
    if self.has_attention_mask or self.has_bool_tensors:
        inputs = list(inputs)
        inputs.pop()
        inputs = tuple(inputs)

    if isinstance(inputs, torch.Tensor):
        assert inputs.grad is not None
        p2p.send(inputs.grad, self.prev_stage)
    else:
        # XXX terrible hacky branch
        if self.is_grad_partitioned:
            # First two sends are partitioned gradient
            p2p.send(inputs[0], self.prev_stage)
            p2p.send(inputs[1], self.prev_stage)
        else:
            for idx, buffer in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not buffer.is_floating_point():
                    assert buffer.grad is None
                    continue
                assert buffer.grad is not None
                p2p.send(buffer.grad, self.prev_stage)

    # We can free up the input buffer now
    self.pipe_buffers['inputs'][buffer_id] = None

    if self.wall_clock_breakdown():
        self.timers('pipe_send_grad').stop()


def _exec_reduce_grads(self):
    self._force_grad_boundary = True
    if self.pipeline_enable_backward_allreduce:
        if self.bfloat16_enabled():
            if self.zero_optimization_stage() < ZeroStageEnum.gradients:
                self._bf16_reduce_grads()
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
    self._force_grad_boundary = False


engine.PipelineEngine.ID_TO_DTYPE = ID_TO_DTYPE
engine.PipelineEngine.DTYPE_TO_ID = DTYPE_TO_ID
engine.PipelineEngine._INSTRUCTION_MAP[schedule.BackwardPass] = _exec_backward_pass
engine.PipelineEngine._INSTRUCTION_MAP[schedule.SendGrad] = _exec_send_grads
engine.PipelineEngine._INSTRUCTION_MAP[schedule.ReduceGrads] = _exec_reduce_grads
engine.PipelineEngine._exec_backward_pass = _exec_backward_pass
engine.PipelineEngine._exec_send_grads = _exec_send_grads
engine.PipelineEngine._exec_reduce_grads = _exec_reduce_grads