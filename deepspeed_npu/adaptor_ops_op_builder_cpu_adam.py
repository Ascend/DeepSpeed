import os
import deepspeed
import torch_npu
from deepspeed.ops.op_builder.builder import TorchCPUOpBuilder


torch_npu_path = os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)))
ascend_path = os.environ.get('ASCEND_TOOLKIT_HOME')


def sources(self):
    return [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc_npu/adam/cpu_adam.cpp')]


def include_paths(self):
    return ['csrc/includes', os.path.join(self.ascend_path, 'include'),
            os.path.join(self.torch_npu_path, 'include')]


def extra_ldflags(self):
    return ['-L' + os.path.join(self.ascend_path, 'lib64'), '-lascendcl',
            '-L' + os.path.join(self.torch_npu_path, 'lib'), '-ltorch_npu']


def cxx_args(self):
    return ['-O3', '-std=c++14', '-g', '-Wno-reorder', '-fopenmp',
            '-L' + os.path.join(self.ascend_path, 'lib64'),
            '-L' + os.path.join(self.torch_npu_path, 'lib')]


def nvcc_args(self):
    return []


deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.torch_npu_path = torch_npu_path
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.ascend_path = ascend_path
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.sources = sources
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.include_paths = include_paths
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.extra_ldflags = extra_ldflags
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.cxx_args = cxx_args
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.nvcc_args = nvcc_args