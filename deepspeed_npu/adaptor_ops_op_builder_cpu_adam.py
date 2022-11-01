import sys
import os
import deepspeed
import torch_npu
from deepspeed.ops.op_builder.builder import TorchCPUOpBuilder


class CPUAdamBuilderNPU(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"
    torch_npu_path = os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)))
    ascend_path = os.environ.get('ASCEND_TOOLKIT_HOME')

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return [os.path.join(os.path.dirname(os.path.abspath(__file__))), 'csrc_npu/adam/cpu_adam.cpp']

    def include_paths(self):
        return ['csrc/includes', os.path.join(self.ascend_path, 'include'),
                os.path.join(self.torch_npu_path, 'include')]

    def extra_ldflags(self):
        return ['-L' + os.path.join(self.ascend_path, 'lib64'), '-lascendcl',
                '-L' + os.path.join(self.torch_npu_path, 'lib'), '-ltorch_npu']

    def cxx_args(self):
        return ['-03', '-std=c++14', '-g', '-Wno-reorder', '-fopenmp',
                '-L' + os.path.join(self.ascend_path, 'lib64'),
                '-L' + os.path.join(self.torch_npu_path, 'lib')]

    def nvcc_args(self):
        return []


for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'CPUAdamBuilder'):
        setattr(v, 'CPUAdamBuilder', CPUAdamBuilderNPU)
