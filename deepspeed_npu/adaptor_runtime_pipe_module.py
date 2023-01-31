import os
import sys
import torch
import torch_npu
import torch.nn as nn
import torch.distributed as dist
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.utils import logger


class PipelineModuleNpu(PipelineModule):

    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint,
                 checkpointable_layers=None):
        """Modules to be parallelized with pipeline parallelism.

        The key constraint that enables pipeline parallelism is the
        representation of the forward pass as a sequence of layers
        and the enforcement of a simple interface between them. The
        forward pass is implicitly defined by the module ``layers``. The key
        assumption is that the output of each layer can be directly fed as
        input to the next, like a ``torch.nn.Sequence``. The forward pass is
        implicitly:

        .. code-block:: python

            def forward(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return x

        .. note::
            Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

        Args:
            layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
            num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
            topology (``deepseed.runtime.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
            loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
            base_seed (int, optional): [description]. Defaults to 1234.
            partition_method (str, optional): [description]. Defaults to 'parameters'.
            activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
            activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        """

        nn.Module.__init__(self)

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(checkpointable_layers, list), "param `checkpointable_layers` must be type of list."

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank != None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})'
                    )
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Construct communicators for pipeline topology
        self._grid = PipelineParallelGrid(process_group=self.world_group,
                                          topology=self._topo)

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)

        self.forward_funcs = []
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        self._build()

        torch.npu.set_device(self.local_rank)
        self.to(f'npu:{self.local_rank}')

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

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
            self.parts = ds_utils.partition_balanced(weights=binary_weights,
                                                     num_parts=num_stages)
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
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__
                       for f in funcs)
        if self.checkpointable_layers is not None:
            return all(f.__class__.__name__ in self.checkpointable_layers for f in funcs)

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)


for k, v in sys.modules.items():
    if 'deepspeed' in k and hasattr(v, 'PipelineModule'):
        setattr(v, 'PipelineModule', PipelineModuleNpu)
        