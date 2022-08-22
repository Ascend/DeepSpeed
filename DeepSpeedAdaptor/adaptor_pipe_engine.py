import torch
from deepspeed.runtime.pipe import engine
from deepspeed.runtime.pipe.engine import _tensor_bytes, PipelineEngine
from deepspeed.runtime.pipe import p2p
from deepspeed.runtime.utils import PartitionedTensor
class PipelineEngineNew(PipelineEngine):
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
        super()._exec_backward_pass(buffer_id)

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], tuple) or isinstance(batch[0], list)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

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
            if self.module.__class__.__name__ == 'GPT2ModelPipe' or self.has_bool_tensors:
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

engine.PipelineEngine = PipelineEngineNew
