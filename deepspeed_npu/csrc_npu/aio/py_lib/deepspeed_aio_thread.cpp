// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/
#include "deepspeed_aio_thread.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int num_bytes,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _num_bytes(num_bytes),
      _validate(validate)
{
    auto dtype = _buffer.dtype();
    auto is_device = at_npu::key::isDeviceTensor(_buffer);
    _cpu_buffer = is_device ? _buffer.to(at::Device(at::kCPU), dtype).pin_memory() : _buffer;

    _contiguous_buffer = _cpu_buffer.contiguous();
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::fini()
{
    auto is_device = at_npu::key::isDeviceTensor(_buffer);
    if (_read_op && is_device) { _buffer.copy_(_cpu_buffer.to(at_npu::key::NativeDeviceType)); }
}

deepspeed_aio_thread_t::deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config)
    : _tid(tid),
      _aio_config(aio_config),
      _aio_ctxt(new aio_context(aio_config._block_size, aio_config._queue_depth)),
      _time_to_exit(false)
{
}

deepspeed_aio_thread_t::~deepspeed_aio_thread_t() {}

void deepspeed_aio_thread_t::run()
{
    while (true) {
        std::shared_ptr<struct io_op_desc_t> next_io_op = nullptr;

        {
            std::unique_lock<std::mutex> lock(_work_sync._mutex);
            _work_sync._cond_var.wait(lock,
                                      [this] { return (!_work_queue.empty() || _time_to_exit); });
            if (!_work_queue.empty()) {
                next_io_op = _work_queue.front();
                _work_queue.pop();
            }
        }

        if (next_io_op) {
            const auto base_offset = next_io_op->_num_bytes * _tid;
            auto mem_buffer = next_io_op->data_ptr();
            // Align the buffer to prevent AIO write failures.
            posix_memalign((void**)&mem_buffer, (size_t)sysconf(_SC_PAGESIZE), next_io_op->_num_bytes);
            std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
                next_io_op->_fd, base_offset, next_io_op->_num_bytes, mem_buffer));

            if (_aio_config._overlap_events) {
                do_aio_operation_overlap(
                    next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            } else {
                do_aio_operation_sequential(
                    next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            }

            {
                std::lock_guard<std::mutex> lock(_complete_sync._mutex);
                _complete_queue.push(next_io_op);
            }
            _complete_sync._cond_var.notify_one();
        }

        if (_time_to_exit) { break; }
    }
}
