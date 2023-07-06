import os
import time
import torch
import torch_npu
import deepspeed


torch_npu_path = os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)))
ascend_path = os.environ.get('ASCEND_TOOLKIT_HOME')


def sources(self):
    root = os.path.dirname(os.path.abspath(__file__))
    source_files = [
        'csrc_npu/aio/py_lib/deepspeed_py_copy.cpp', 'csrc_npu/aio/py_lib/py_ds_aio.cpp',
        'csrc_npu/aio/py_lib/deepspeed_py_aio.cpp', 'csrc_npu/aio/py_lib/deepspeed_py_aio_handle.cpp',
        'csrc_npu/aio/py_lib/deepspeed_aio_thread.cpp', 
        'csrc_npu/aio/common/deepspeed_aio_utils.cpp',
        'csrc_npu/aio/common/deepspeed_aio_common.cpp', 
        'csrc_npu/aio/common/deepspeed_aio_types.cpp',
        'csrc_npu/aio/py_lib/deepspeed_pin_tensor.cpp'
    ]
    all_sources = [os.path.join(root, i) for i in source_files]
    return all_sources


def include_paths(self):
    root = os.path.dirname(os.path.abspath(__file__))
    include_paths = [
        os.path.join(root, 'csrc_npu/aio/py_lib'), 
        os.path.join(root, 'csrc_npu/aio/common'), 
        os.path.join(self.ascend_path, 'include'),
        os.path.join(self.torch_npu_path, 'include')
        ]
    return include_paths


deepspeed.ops.op_builder.async_io.AsyncIOBuilder.ascend_path = ascend_path
deepspeed.ops.op_builder.async_io.AsyncIOBuilder.torch_npu_path = torch_npu_path
deepspeed.ops.op_builder.async_io.AsyncIOBuilder.sources = sources
deepspeed.ops.op_builder.async_io.AsyncIOBuilder.include_paths = include_paths