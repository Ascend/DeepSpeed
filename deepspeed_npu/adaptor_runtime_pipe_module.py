import os
import sys
import torch
import torch_npu
import deepspeed
import deepspeed.runtime.utils as ds_utils

from deepspeed.utils import logger
from deepspeed.runtime.pipe.module import LayerSpec


def _partition_layers(self, method='uniform'):
    num_stages = self._topo.get_dim('pipe')
    stage_id = self._topo.get_coord(self.global_rank).pipe

    if self.global_rank == 0:
        logger.info(f'Partitioning pipeline stages with method {method}')

    method = method.lower()

    # Each stage gets a simple uniform number of layers.
    if method == 'uniform':
        num_layers = len(self._layer_specs)
        self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
    elif method == 'parameters':
        param_counts = self._count_layer_params()
        self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
    elif method == 'greedy':
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
    elif method.startswith('type:'):
        layertype = method.split(':')[1]
        binary_weights = [0] * len(self._layer_specs)
        for idx in self._find_layer_type(layertype):
            binary_weights[idx] = 1
        self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
    elif method == 'profile':
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')

    # Print some information on the partitioning.
    if self.global_rank == 0:
        for stage in range(num_stages):
            start = self.parts[stage]
            stop = self.parts[stage + 1]
            print(f'stage={stage} layers={stop - start}')
            for idx, layer in enumerate(self._layer_specs[start:stop]):
                name = str(layer)
                if isinstance(layer, LayerSpec):
                    name = layer.typename.__name__
                if isinstance(layer, torch.nn.Module):
                    name = layer.__class__.__name__
                else:
                    try:
                        name = layer.__name__
                    except AttributeError:
                        pass
                print(f'    {idx+start:2d}: {name}')
        if self.loss_fn:
            try:
                print(f'  loss: {self.loss_fn.__name__}')
            except AttributeError:
                print(f'  loss: {self.loss_fn.__class__.__name__}')

    self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])


def _is_checkpointable(self, funcs):
    # This is an unfortunate hack related to torch and deepspeed activation checkpoint implementations.
    # Some layers like torch.nn.Embedding will not receive grads if checkpointed, which breaks things.
    # I presume it's related to the discrete inputs that cannot require_grad? Need to revisit.
    if self.__class__.__name__ in ('T5Pipeline'):
        return all('T5BlockPipeline' in f.__class__.__name__ for f in funcs)
    if self.__class__.__name__ in ('GPTModelPipe', 'GPT2ModelPipe'):
        return all('ParallelTransformerLayerPipe' in f.__class__.__name__ for f in funcs)
    if self.checkpointable_layers is not None:
        return all(f.__class__.__name__ in self.checkpointable_layers for f in funcs)

    params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
    return any(len(list(p)) > 0 for p in params)

deepspeed.runtime.pipe.module.PipelineModule._partition_layers = _partition_layers
deepspeed.runtime.pipe.module.PipelineModule._is_checkpointable = _is_checkpointable


