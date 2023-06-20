import sys
import torch
import torch_npu
from torch_npu.npu import clear_npu_overflow_flag
from deepspeed.runtime.pipe.engine import _tensor_bytes
from deepspeed.runtime.pipe import p2p, engine, schedule
from deepspeed.runtime.utils import PartitionedTensor

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


def _exec_backward_pass(self, buffer_id):
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


def _send_tensor_meta(self, buffer, recv_stage):
    """ Communicate metadata about upcoming p2p transfers.

    Metadata is communicated in this order:
        * type (0: tensor, 1: list)
        * num_tensors if type=list
        foreach tensor in buffer:
            * ndims
            * shape
    """
    send_bytes = 0
    if isinstance(buffer, torch.Tensor):
        type_tensor = torch.IntTensor(data=[0]).to(self.device)
        p2p.send(type_tensor, recv_stage)
        send_shape = torch.IntTensor(data=buffer.size()).to(self.device)
        send_ndims = torch.IntTensor(data=[len(buffer.size())]).to(self.device)
        p2p.send(send_ndims, recv_stage)
        p2p.send(send_shape, recv_stage)
        send_bytes += _tensor_bytes(buffer)
    elif isinstance(buffer, list):
        assert (False)
        type_tensor = torch.IntTensor(data=[1]).to(self.device)
        p2p.send(type_tensor, recv_stage)
        count_tensor = torch.IntTensor(data=[len(buffer)]).to(self.device)
        p2p.send(count_tensor, recv_stage)
        for tensor in buffer:
            assert isinstance(tensor, torch.Tensor)
            send_shape = torch.IntTensor(data=tensor.size()).to(self.device)
            send_ndims = torch.IntTensor(data=[len(tensor.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(tensor)
    elif isinstance(buffer, tuple):
        type_tensor = torch.IntTensor(data=[2]).to(self.device)
        p2p.send(type_tensor, recv_stage)
        count_tensor = torch.IntTensor(data=[len(buffer)]).to(self.device)
        p2p.send(count_tensor, recv_stage)
        for idx, tensor in enumerate(buffer):
            assert isinstance(tensor, torch.Tensor)
            send_shape = torch.IntTensor(data=tensor.size()).to(self.device)
            send_ndims = torch.IntTensor(data=[len(tensor.size())]).to(self.device)
            send_dtype = torch.IntTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(
                self.device)
            p2p.send(send_dtype, recv_stage)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            # Useful for performance debugging.
            '''
            new_bytes = _tensor_bytes(tensor)
            send_bytes += _tensor_bytes(tensor)
            # Useful for performance debugging.
            if self.grid.data_parallel_id == 0:
                print(
                    f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                )
            '''
    else:
        raise NotImplementedError(f'Could not send meta type {type(buffer)}')

    # Useful for performance debugging.
    '''
    if self.grid.data_parallel_id == 0:
        print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
    '''


def _recv_tensor_meta(self, send_stage):
    """Receive metadata about upcoming p2p transfers and return allocated buffers.

    Metadata is communicated in this order:
        * type (0: tensor, 1: list)
        * num_tensors if type=list
        foreach tensor in buffer:
            * ndims
            * shape

    Returns:
        Allocated buffer for receiving from send_stage.
    """

    type_tensor = torch.IntTensor(data=[0]).to(self.device)
    p2p.recv(type_tensor, send_stage)
    recv_type = type_tensor.item()

    # A single tensor will be sent.
    if recv_type == 0:
        recv_ndims = torch.IntTensor(data=[0]).to(self.device)
        p2p.recv(recv_ndims, send_stage)
        recv_ndims = recv_ndims.item()
        recv_shape = torch.IntTensor([1] * recv_ndims).to(self.device)
        p2p.recv(recv_shape, send_stage)
        recv_shape = recv_shape.tolist()
        return self._allocate_buffer(recv_shape, num_buffers=1)[0]

    # List or tuple of tensors
    elif recv_type == 1 or recv_type == 2:
        count_tensor = torch.IntTensor(data=[0]).to(self.device)
        p2p.recv(count_tensor, send_stage)
        num_tensors = count_tensor.item()
        recv_shapes_and_dtypes = []
        for idx in range(num_tensors):
            recv_dtype = torch.IntTensor(data=[0]).to(self.device)
            p2p.recv(recv_dtype, send_stage)
            recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
            recv_ndims = torch.IntTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.IntTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

        buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
        # Convert to tuples if requested.
        if recv_type == 2:
            buffers = tuple(buffers)
        return buffers

    else:
        raise NotImplementedError(f'Could not receive type {type(recv_type)}')


def _exec_recv_activations(self, buffer_id):
    if self.wall_clock_breakdown():
        self.timers('pipe_recv_input').start()

    recvd = None

    # Allocate the buffer if necessary
    if self.pipe_recv_buf is None:
        self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

    if isinstance(self.pipe_recv_buf, torch.Tensor):
        p2p.recv(self.pipe_recv_buf, self.prev_stage)
        recvd = self.pipe_recv_buf.clone().detach()
        recvd.requires_grad = recvd.is_floating_point()
    else:
        assert isinstance(self.pipe_recv_buf, tuple)
        recvd = [None] * len(self.pipe_recv_buf)
        for idx, buffer in enumerate(self.pipe_recv_buf):
            assert torch.is_tensor(buffer)
            # XXX hardcode meta type
            if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.int:
                if self.meta_buffer is None:
                    self.meta_buffer = torch.zeros(buffer.size(),
                                                   dtype=torch.int,
                                                   device=self.device)
                buffer = self.meta_buffer

            p2p.recv(buffer, self.prev_stage)
            recvd[idx] = buffer.clone().detach()

        # NCCL does not like to send torch.BoolTensor types, so un-cast the
        # attention mask
        if self.has_attention_mask or self.has_bool_tensors:
            recvd[-1] = recvd[-1].bool()

        recvd = tuple(recvd)

        for buffer in recvd:
            buffer.requires_grad = buffer.is_floating_point()

    self.pipe_buffers['inputs'][buffer_id] = recvd

    if self.wall_clock_breakdown():
        self.timers('pipe_recv_input').stop()


def _exec_recv_grads(self, buffer_id):
    if self.wall_clock_breakdown():
        self.timers('pipe_recv_grad').start()

    outputs = self.pipe_buffers['outputs'][buffer_id]
    # XXX these shapes are hardcoded for Megatron
    # Restore partitioned output if it was partitioned and we are sending full gradients
    if self.is_pipe_partitioned and not self.is_grad_partitioned:
        part_output = PartitionedTensor.from_meta(
            meta=outputs[0],
            local_part=outputs[1],
            group=self.grid.get_slice_parallel_group())
        outputs[0].data = part_output.full()
        outputs = (outputs[0], *outputs[2:])
        # save for backward
        self.pipe_buffers['outputs'][buffer_id] = outputs

    # Allocate gradient if necessary
    if self.grad_layer is None:
        if isinstance(outputs, torch.Tensor):
            s = list(outputs.size())
            self.grad_layer = self._allocate_buffer(s,
                                                    dtype=outputs.dtype,
                                                    num_buffers=1)[0]
        else:
            # XXX This is a HACK
            # When we exchange activations/gradients, the two pipe stages
            # need to issue the send/recv with the same buffer sizes or
            # else there is a deadlock. The is_floating_point() filter is
            # used to avoid sending gradients for tensors that do not
            # produce gradients. When TP>1, we partition the first
            # activations/gradients across TP ranks to save communication
            # volume and memory. That partitioned tensor is represented as
            # two tensors: a 1/TPth chunk of the original data and also a
            # small LongTensor storing the metadata used to reconstruct on
            # the other side. When combined, the floating point filter also
            # filtered out the metadata tensor. This quick (hacky) fix just
            # branches on is_grad_partitioned so we don't filter out the
            # metadata tensor.
            if self.is_grad_partitioned:
                sizes_and_dtypes = [
                                       (list(t.size()),
                                        t.dtype) for t in outputs[:2]
                                   ] + [(list(t.size()),
                                         t.dtype) for t in outputs[2:] if t.is_floating_point()]
            else:
                sizes_and_dtypes = [(list(t.size()),
                                     t.dtype) for t in outputs
                                    if t.is_floating_point()]
            self.grad_layer = self._allocate_buffers(sizes_and_dtypes,
                                                     num_buffers=1)[0]

    if isinstance(self.grad_layer, torch.Tensor):
        p2p.recv(self.grad_layer, self.next_stage)
    else:
        assert isinstance(outputs, tuple)
        for idx, buffer in enumerate(self.grad_layer):
            # XXX GPT-2 hack
            if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.int:
                buffer.data = torch.zeros(buffer.size(),
                                          dtype=torch.int,
                                          device=self.device)
            p2p.recv(buffer, self.next_stage)

    if self.wall_clock_breakdown():
        self.timers('pipe_recv_grad').stop()


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
            # DeepSpeed bug? PR https://github.com/microsoft/DeepSpeed/pull/2538
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


engine.PipelineEngine.ID_TO_DTYPE = ID_TO_DTYPE
engine.PipelineEngine._INSTRUCTION_MAP[schedule.BackwardPass] = _exec_backward_pass
engine.PipelineEngine._INSTRUCTION_MAP[schedule.RecvActivation] = _exec_recv_activations
engine.PipelineEngine._INSTRUCTION_MAP[schedule.RecvGrad] = _exec_recv_grads
engine.PipelineEngine._INSTRUCTION_MAP[schedule.SendGrad] = _exec_send_grads
engine.PipelineEngine._exec_backward_pass = _exec_backward_pass
engine.PipelineEngine._send_tensor_meta = _send_tensor_meta
engine.PipelineEngine._recv_tensor_meta = _recv_tensor_meta
engine.PipelineEngine._exec_recv_activations = _exec_recv_activations
engine.PipelineEngine._exec_recv_grads = _exec_recv_grads
engine.PipelineEngine._exec_send_grads = _exec_send_grads