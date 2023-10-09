import os
import torch_npu
import deepspeed


def sources(self):
    root = os.path.dirname(os.path.abspath(deepspeed.__file__))
    root_npu = os.path.dirname(os.path.abspath(__file__))

    source_files = [
        'ops/csrc/aio/py_lib/deepspeed_py_copy.cpp', 'ops/csrc/aio/py_lib/py_ds_aio.cpp',
        'ops/csrc/aio/py_lib/deepspeed_py_aio.cpp', 'ops/csrc/aio/py_lib/deepspeed_py_aio_handle.cpp',
        'ops/csrc/aio/common/deepspeed_aio_utils.cpp',
        'ops/csrc/aio/common/deepspeed_aio_common.cpp', 'ops/csrc/aio/common/deepspeed_aio_types.cpp',
        'ops/csrc/aio/py_lib/deepspeed_pin_tensor.cpp'
    ]

    all_sources = [os.path.join(root, i) for i in source_files]

    source_files = ['csrc_npu/aio/py_lib/deepspeed_aio_thread.cpp']
    all_sources += [os.path.join(root_npu, i) for i in source_files]
    return all_sources


def include_paths(self):
    root = os.path.dirname(os.path.abspath(deepspeed.__file__))
    torch_npu_path = os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)))

    include_paths_list = [
        os.path.join(root, 'ops/csrc/aio/py_lib'),
        os.path.join(root, 'ops/csrc/aio/common'),
        os.path.join(torch_npu_path, 'include')
    ]
    return include_paths_list


def cxx_args(self):
    # -O0 for improved debugging, since performance is bound by I/O
    CPU_ARCH = self.cpu_arch()
    SIMD_WIDTH = self.simd_width()
    args = ['-g', '-Wall', '-O0', '-std=c++14', '-shared', '-Wno-reorder', CPU_ARCH, '-fopenmp', SIMD_WIDTH,
            '-laio']
    args += ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack',
             '-fPIC -pie', '-Wl,--disable-new-dtags,--rpath']
    return args


deepspeed.ops.op_builder.async_io.AsyncIOBuilder.sources = sources
deepspeed.ops.op_builder.async_io.AsyncIOBuilder.include_paths = include_paths
