import sys
import torch
import torch_npu
from deepspeed.runtime.pipe.module import PipelineModule
import deepspeed.runtime.utils as ds_utils
from deepspeed.utils import logger
class PipelineModuleNpu(PipelineModule):
    def _index_tied_modules(self):
        torch.npu.set_device(self.local_rank)
        self.to(f'npu:{self.local_rank}')
        super()._index_tied_modules()

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts,
                                                     num_parts=num_stages)
        elif method == 't5':
            param_counts = self._count_layer_params()

            def partition(weights, num_parts):
                if len(weights) <= num_parts:
                    return ds_utils.partition_uniform(len(weights), num_parts)
                parts = list(range(len(weights) + 1))

                weights_ = weights[:]
                while len(parts) > num_parts + 1:
                    mini = weights_.index(min(weights_))

                    left = right = float('inf')
                    if mini - 1 >= 0:
                        left = weights_[mini - 1]
                    if mini + 1 < len(weights_):
                        right = weights_[mini + 1]
                    if left <= right:
                        weights_[mini] += left
                        weights_.pop(mini - 1)
                        parts.pop(mini)
                    else:
                        weights_[mini] += right
                        weights_.pop(mini + 1)
                        parts.pop(mini + 1)

                return parts

            self.parts = partition(param_counts, num_stages)

for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'PipelineModule'):
        setattr(v, 'PipelineModule', PipelineModuleNpu)
        