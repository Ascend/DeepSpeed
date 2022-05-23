import math
from typing import Dict, List, Set
import pytest
import torch.distributed as dist
import torch
if "1.5.0" not in torch.__version__:
    import torch_npu
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.loss import L1Loss
from torch.nn.parameter import Parameter

from .common import distributed_test
from .simple_model import SimpleModel, random_dataloader, args_from_dict

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def run_unbalanced_gradients(model, data_loader):
    def drop_some_gradients(model, iter):
        odd_iteration = iter % 2
        for i, p in enumerate(model.parameters()):
            p.requires_grad = (i % 2) == odd_iteration

    def enable_grads(model):
        for p in model.parameters():
            p.requires_grad = True

    for i, batch in enumerate(data_loader):
        drop_some_gradients(model, i + 1)
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()
        enable_grads(model)


def dump_state_dict(model):
    if dist.get_rank() == 0:
        print("state_dict:")
        for name, param in model.named_parameters():
            print(f"{name} {param.data}")


@pytest.mark.parametrize('zero_stage', [1, 2, 3])
def test_zero_unbalanced_gradients(tmpdir, zero_stage):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                # ASCEND AVOID
                "torch_adam": True,
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[2])
    def _test_zero_unbalanced_gradients(model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        run_unbalanced_gradients(model, data_loader)

    _test_zero_unbalanced_gradients(model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
@pytest.mark.parametrize('zero_stage', [3])
def test_zero3_repeat_forward_loop(tmpdir, zero_stage):

    # force all params to be partitioned by forcing threshold=0
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 0
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    hidden_dim = 4

    class AlbertLikeModel(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, y):
            # run the same layer multiple times in a loop - to test a stack of forwards, followed by a stack of backwards
            hidden = x
            for i in range(3):
                hidden = hidden + self.linear(hidden)
            return self.cross_entropy_loss(hidden, y)

    model = AlbertLikeModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_zero3_repeat_forward_loop(model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero3_repeat_forward_loop(model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
# also reproduces the https://github.com/microsoft/DeepSpeed/pull/1372
@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32_1_param_group(tmpdir, zero_stage):

    # XXX: ideally refactor with the 2_param_group test as 75% is the same

    # force all params to be partitioned by forcing threshold=0
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 0
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    @distributed_test(world_size=[2])
    def _test_zero_to_fp32():
        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim, n_layers):
                super().__init__()
                # to reproduce https://github.com/microsoft/DeepSpeed/pull/1372 it is important that
                # the number of total elements is uneven:
                # (1) 4 layers of 3*(3+1)=12 elements each, 48 in total
                self.ll = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim,
                                    hidden_dim) for i in range(n_layers))
                # (2) the following adds 4+1=5 elements
                self.classifier = torch.nn.Linear(4, 1)
                # total 48+5=53 (uneven as desired) elements
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        hidden_dim = 3  # do not change

        world_size = dist.get_world_size()
        # we want at least 2x layers as there are gpus to trigger round_robin_fp16_groups reshuffle in zero2
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers)

        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()

        if dist.get_rank() == 0:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            #dump_state_dict(fp32_model)

            fp32_state_dict = fp32_model.state_dict()
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(),
                                      fp32_state_dict[name].float())

    _test_zero_to_fp32()


@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32_2_param_groups(tmpdir, zero_stage):

    # TODO:
    # - need to test with multiple param groups

    # force all params to be partitioned by forcing threshold=0
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_allow_untested_optimizer": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_param_persistence_threshold": 0
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    @distributed_test(world_size=[2])
    def _test_zero_to_fp32():
        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim, n_layers):
                super().__init__()
                self.ll = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim,
                                    hidden_dim) for i in range(n_layers))
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        hidden_dim = 3

        world_size = dist.get_world_size()
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers)

        optim_groups = [
            {
                "params": [l.weight for l in model.ll],
                "weight_decay": 0.01,
            },
            {
                "params": [l.bias for l in model.ll],
                "weight_decay": 0.0
            },
        ]
        optim = torch.optim.SGD(optim_groups, lr=0.1)

        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              optimizer=optim,
                                              config=config_dict
        )
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()

        if dist.get_rank() == 0:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            #dump_state_dict(fp32_model)

            fp32_state_dict = fp32_model.state_dict()
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(),
                                      fp32_state_dict[name].float())

    _test_zero_to_fp32()


@pytest.mark.parametrize('zero_stage, allgather_bucket_size', [(2, 1000), (2, 1001)])
def test_incorrect_allgather_bucket_size(tmpdir, zero_stage, allgather_bucket_size):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_bucket_size": allgather_bucket_size
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_incorrect_allgather_bucket_size(model, hidden_dim):
        if allgather_bucket_size % 2 == 0:
            model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        else:
            with pytest.raises(AssertionError) as assertinfo:
                model, _, _, _ = deepspeed.initialize(config=config_dict,
                                                  model=model,
                                                  model_parameters=model.parameters())
            assert "allgather_bucket_size must be a multiple of nccl_start_alignment_factor" in str(
                assertinfo)

    _test_incorrect_allgather_bucket_size(model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, world_size', [(2, 2), (2, 3), (2, 4)])
def test_partition_nccl_alignment(tmpdir, zero_stage, world_size):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=world_size)
    def _test_partition_nccl_alignment(model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())

        # get nccl all-gather send buffers alignment factor
        nccl_start_alignment_factor = model.optimizer.nccl_start_alignment_factor

        parallel_partitioned_bit16_groups = model.optimizer.parallel_partitioned_bit16_groups if zero_stage == 2 else model.optimizer.parallel_partitioned_fp16_groups
        for data_parallel_partitions in parallel_partitioned_bit16_groups:
            for partition_id, partitioned_data in enumerate(data_parallel_partitions):
                # verify that data partition start locations are 4-byte aligned
                assert (partitioned_data.data_ptr() %
                        (2 * nccl_start_alignment_factor) == 0)

    _test_partition_nccl_alignment(model=model, hidden_dim=hidden_dim)


def _ds_initialize_for_param_partitioning_testing(model: Module,
                                                  cfg: dict) -> DeepSpeedEngine:
    ds_engine, _, _, _ = deepspeed.initialize(
        config=cfg,
        model=model,
        model_parameters=model.parameters()
    )

    return ds_engine


def _assert_partition_status(model: Module,
                             valid_statuses: Set[ZeroParamStatus]) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status in valid_statuses, param.ds_summary()


def _assert_fully_available(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status == ZeroParamStatus.AVAILABLE


class EltwiseMultiplicationModule(Module):
    def __init__(self, weight: Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        _assert_fully_available(self)
        result = self.weight * x

        return result


class EltwiseMultiplicationTestNetwork(Module):
    """used for testing purposes"""
    def __init__(
        self,
        weight1: Parameter,
        weight2: Parameter,
        weight3: Parameter,
    ) -> None:
        super().__init__()
        self.__layer1 = EltwiseMultiplicationModule(weight1)
        self.__layer2 = EltwiseMultiplicationModule(weight2)
        self.__layer3 = EltwiseMultiplicationModule(weight3)

        self.loss = L1Loss(reduction="none")

    def forward(self, x: Tensor, y: Tensor, prefetching: bool) -> Dict[str, Tensor]:
        _assert_partition_status(
            self,
            {
                ZeroParamStatus.NOT_AVAILABLE,
                ZeroParamStatus.INFLIGHT,
                ZeroParamStatus.AVAILABLE
            } if prefetching else {ZeroParamStatus.NOT_AVAILABLE})

        layerwise_expected_states = {
            ZeroParamStatus.INFLIGHT if prefetching else ZeroParamStatus.NOT_AVAILABLE,
            ZeroParamStatus.AVAILABLE,
        }

        _assert_partition_status(self.__layer1, layerwise_expected_states)
        hidden1 = self.__layer1(x)
        _assert_partition_status(self.__layer1, {ZeroParamStatus.NOT_AVAILABLE})

        _assert_partition_status(self.__layer2, layerwise_expected_states)
        hidden2 = self.__layer2(hidden1)
        _assert_partition_status(self.__layer2, {ZeroParamStatus.NOT_AVAILABLE})

        _assert_partition_status(self.__layer3, layerwise_expected_states)
        y_hat = self.__layer3(hidden2)
        _assert_partition_status(self.__layer3,
                                 {
                                     ZeroParamStatus.AVAILABLE
                                     if prefetching else ZeroParamStatus.NOT_AVAILABLE
                                 })

        loss = self.loss(y_hat, y)

        _assert_partition_status(
            self,
            {
                ZeroParamStatus.NOT_AVAILABLE,
                ZeroParamStatus.INFLIGHT,
                ZeroParamStatus.AVAILABLE
            } if prefetching else {ZeroParamStatus.NOT_AVAILABLE})

        return {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "y_hat": y_hat,
            "loss": loss,
        }


@pytest.mark.parametrize("param_persistence_threshold", [0, 10])
@pytest.mark.parametrize("fp16_enabled", [True, False])
@pytest.mark.parametrize("contiguous_gradients", [True, False])
@pytest.mark.parametrize("offload_optimizer", [True, False])
@pytest.mark.parametrize("zero_grad", [True, False])
@pytest.mark.parametrize("iteration", list(range(1)))
def test_zero3_param_partitioning_base(
    param_persistence_threshold: int,
    fp16_enabled: bool,
    contiguous_gradients: bool,
    offload_optimizer: bool,
    zero_grad: bool,
    iteration: int,
) -> None:
    @distributed_test(world_size=[2])
    def _test_zero3_param_partitioning():
        if offload_optimizer and not contiguous_gradients:
            return

        m = 3
        n = 5
        weights = [Parameter(torch.zeros((m, n), dtype=torch.float32)) for _ in range(3)]
        model = EltwiseMultiplicationTestNetwork(*weights)

        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "stage3_param_persistence_threshold": param_persistence_threshold,
                "contiguous_gradients": contiguous_gradients,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": fp16_enabled,
                "loss_scale": 1.,
            }
        }

        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, cfg)
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(weight.ds_tensor.data,
                                                    (i + 1) * (1 + dist.get_rank()))

        def create_tensor(vals, dtype: torch.dtype = None) -> Tensor:
            return torch.as_tensor(vals,
                                   dtype=dtype
                                   or (torch.float16 if fp16_enabled else torch.float32),
                                   device=ds_engine.device)

        expected_hidden1 = create_tensor([
            [1,
             1,
             1,
             1,
             1],
            [1,
             1,
             1,
             2,
             2],
            [2,
             2,
             2,
             2,
             2],
        ])
        expected_hidden2 = create_tensor([
            [2,
             2,
             2,
             2,
             2],
            [2,
             2,
             2,
             8,
             8],
            [8,
             8,
             8,
             8,
             8],
        ])
        expected_yhat = create_tensor([[6,
                                        6,
                                        6,
                                        6,
                                        6],
                                       [6,
                                        6,
                                        6,
                                        48,
                                        48],
                                       [48,
                                        48,
                                        48,
                                        48,
                                        48]])
        expected_loss = create_tensor([
            [5,
             5,
             5,
             5,
             5],
            [5,
             5,
             5,
             47,
             47],
            [47,
             47,
             47,
             47,
             47],
        ])

        for train_iter in range(3):
            activations = ds_engine(
                x=torch.ones((m,
                              n),
                             dtype=torch.float16 if fp16_enabled else torch.float32,
                             device=ds_engine.device),
                y=torch.ones((m,
                              n),
                             dtype=torch.float16 if fp16_enabled else torch.float32,
                             device=ds_engine.device),
                prefetching=train_iter > 0,
            )
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())

            # check the gradients
            grad_partitions = ds_engine.optimizer.get_fp32_grad_partitions()
            assert set(grad_partitions.keys()) == {0}, f"should have one parameter group but got {len(grad_partitions)}"
            assert set(grad_partitions[0].keys()) == {0, 1, 2}
            dloss_wrt_layer1 = grad_partitions[0][0]
            dloss_wrt_layer2 = grad_partitions[0][1]
            dloss_wrt_layer3 = grad_partitions[0][2]

            assert dloss_wrt_layer1.dtype == torch.float
            assert dloss_wrt_layer2.dtype == torch.float
            assert dloss_wrt_layer3.dtype == torch.float

            # layer1 = [..., 1, 2, ...]
            # layer2 = [..., 2, 4, ...]
            # layer3 = [..., 3, 6, ...]
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x

            grad_multiplier = 1 if zero_grad else (train_iter + 1)
            if dist.get_rank() == 0:
                assert torch.allclose(
                    dloss_wrt_layer3.cuda(),
                    grad_multiplier * create_tensor([2] * 8,
                                                    torch.float))
                assert torch.allclose(
                    dloss_wrt_layer2.cuda(),
                    grad_multiplier * create_tensor([3 * 1] * 8,
                                                    torch.float))
                assert torch.allclose(
                    dloss_wrt_layer1.cuda(),
                    grad_multiplier * create_tensor([3 * 2 * 1] * 8,
                                                    torch.float))
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(
                    dloss_wrt_layer3.cuda(),
                    grad_multiplier * create_tensor(([8] * 7) + [0],
                                                    torch.float))
                assert torch.allclose(
                    dloss_wrt_layer2.cuda(),
                    grad_multiplier * create_tensor(([6 * 2] * 7) + [0],
                                                    torch.float))
                assert torch.allclose(
                    dloss_wrt_layer1.cuda(),
                    grad_multiplier * create_tensor(([6 * 4 * 1] * 7) + [0],
                                                    torch.float))
            else:
                raise RuntimeError("test has world size of two")

            if zero_grad:
                ds_engine.optimizer.zero_grad()

        # TODO. add testing for this - for now we just call it to make sure it
        # doesn't throw
        ds_engine.optimizer.step()
        # taking an optimizer step invalidates all parameters, make sure everything
        # has been partitioned afterwards
        _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
        assert not math.isclose(ds_engine.optimizer._global_grad_norm, 0.0)

    _test_zero3_param_partitioning()


@pytest.mark.parametrize("world_sz", [1, 2, 4])
@pytest.mark.parametrize("param_sz", [8100])
@pytest.mark.parametrize("init_context_manager", [True, False])
def test_zero3_param_partitioning_large_param(world_sz: int,
                                              param_sz: int,
                                              init_context_manager: bool) -> None:
    class LargeParamModel(Module):
        def __init__(self):
            super().__init__()
            self.param = Parameter(torch.zeros((param_sz, ), dtype=torch.float32))

            # only do weight initialization on root rank to
            # make sure we are broadcasting correctly from rank 0
            if dist.get_rank() == 0:
                partition_sz = math.ceil(self.param.numel() / dist.get_world_size())
                offset = 0
                for rank in range(dist.get_world_size()):
                    with torch.no_grad():
                        self.param[offset:offset + partition_sz].fill_(rank)
                    offset += partition_sz

        def forward(self, x: Tensor) -> Tensor:
            return x * self.param

    @distributed_test(world_size=[world_sz])
    def _distributed_test():
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }
        with deepspeed.zero.Init(mem_efficient_linear=False,
                                 enabled=init_context_manager):
            model = LargeParamModel()
        ds_engine = _ds_initialize_for_param_partitioning_testing(model, ds_config)

        for train_iter in range(3):  # test multiple iterations to cover prefetching
            activation: Tensor = ds_engine(
                torch.ones(param_sz,
                           dtype=torch.float16,
                           device=ds_engine.device))

            partition_sz = math.ceil(param_sz / world_sz)
            for rank_idx, start_idx in enumerate(range(0, param_sz, partition_sz)):
                activation_from_partition = activation[start_idx:start_idx +
                                                       partition_sz]
                assert torch.allclose(
                    activation_from_partition,
                    torch.full_like(activation_from_partition,
                                    rank_idx))

            ds_engine.backward(activation.sum())
            ds_engine.allreduce_gradients()

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradient, = avgd_gradients[0]
            expected_weight_gradient = (train_iter + 1) * torch.full_like(
                weight_gradient,
                1)

            assert torch.allclose(weight_gradient, expected_weight_gradient)

    _distributed_test()


@pytest.mark.parametrize("world_sz", [1, 2, 4])
@pytest.mark.parametrize("param_sz", [100, 1_000, 10_000])
@pytest.mark.parametrize("n_layers", [100, 1_000])
@pytest.mark.parametrize("init_context_manager", [True, False])
def test_zero3_param_partitioning_many_params(world_sz: int,
                                              param_sz: int,
                                              n_layers: int,
                                              init_context_manager: bool) -> None:
    class ManyParamModel(Module):
        def __init__(self) -> None:
            super().__init__()

            self.modulelist = ModuleList(
                EltwiseMultiplicationModule(
                    weight=Parameter(torch.empty((param_sz,
                                                  ),
                                                 dtype=torch.float32)))
                for _ in range(n_layers))

            for layer_num, module in enumerate(self.modulelist):
                with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                    param: Parameter = module.weight
                    partition_sz = math.ceil(param.numel() / dist.get_world_size())
                    offset = 0
                    for rank in range(dist.get_world_size()):
                        with torch.no_grad():
                            param[offset:offset + partition_sz].fill_(2 * layer_num *
                                                                      rank)
                        offset += partition_sz

        def forward(self, x: Tensor) -> Tensor:
            activations = []

            for module in self.modulelist:
                x = module(x)
                activations.append(x)

            return activations

    @distributed_test(world_size=[world_sz])
    def _distributed_test():
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        with deepspeed.zero.Init(config=ds_cfg,
                                 mem_efficient_linear=False,
                                 enabled=init_context_manager):
            model = ManyParamModel()

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, ds_cfg)

        for _ in range(3):  # test multiple iterations to cover prefetching
            activations: List[Tensor] = ds_engine(
                torch.ones((param_sz,
                            ),
                           dtype=torch.float16,
                           device=ds_engine.device))
            assert len(activations) == n_layers

            partition_sz = math.ceil(param_sz / world_sz)
            expected_activations = torch.empty(param_sz,
                                               dtype=torch.float16,
                                               device=ds_engine.device)
            for start_idx in range(0, param_sz, partition_sz):
                expected_activations[start_idx:start_idx +
                                     partition_sz] = dist.get_rank()

            for layer_num, activation in enumerate(activations):
                expected_activations *= 2 * layer_num
                assert torch.allclose(activation, expected_activations)

            # TODO. finish writing this test
            ds_engine.backward(activations[-1].sum())

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradients: List[Tensor] = avgd_gradients[0]

            for layer_num, activation in enumerate(weight_gradients):
                pass

    _distributed_test()


@pytest.mark.parametrize("world_sz", [1, 2, 4])
def test_zero3_init_for_parent_weight_initialization(world_sz):
    class ModelWhereParentInitializesChildWeights(Module):
        def __init__(self) -> None:
            super().__init__()

            self.linear = Linear(12, 1)

            self.apply(self.__init_weights)

        def __init_weights(self, module):
            if isinstance(module, Linear):
                with torch.no_grad():
                    module.weight.fill_(1 + dist.get_rank())

    @distributed_test(world_size=[world_sz])
    def _distributed_test():
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        with deepspeed.zero.Init(config=ds_cfg,
                                 mem_efficient_linear=False,
                                 enabled=True):
            model = ModelWhereParentInitializesChildWeights()

        assert model.linear.weight.ds_tensor.numel() == math.ceil(12 / world_sz)
        assert torch.allclose(model.linear.weight.ds_tensor,
                              torch.full_like(model.linear.weight.ds_tensor,
                                              1))

    _distributed_test()


@pytest.mark.skip(
    reason="depends on upgraded pytorch and nccl that isn't always available")
@pytest.mark.parametrize("param_persistence_threshold", [0, 10])
@pytest.mark.parametrize("contiguous_gradients", [True, False])
@pytest.mark.parametrize("offload_optimizer", [True, False])
@pytest.mark.parametrize("zero_grad", [True])
@pytest.mark.parametrize("iteration", list(range(1)))
def test_zero3_param_partitioning_base_bf16(
    param_persistence_threshold: int,
    contiguous_gradients: bool,
    offload_optimizer: bool,
    zero_grad: bool,
    iteration: int,
) -> None:
    @distributed_test(world_size=[2])
    def _test_zero3_param_partitioning():
        if offload_optimizer and not contiguous_gradients:
            return

        m = 3
        n = 5
        weights = [Parameter(torch.zeros((m, n), dtype=torch.float32)) for _ in range(3)]
        model = EltwiseMultiplicationTestNetwork(*weights)

        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "stage3_param_persistence_threshold": param_persistence_threshold,
                "contiguous_gradients": contiguous_gradients,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "bf16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, cfg)
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(weight.ds_tensor.data,
                                                    (i + 1) * (1 + dist.get_rank()))

        def create_tensor(vals):
            return torch.as_tensor(vals, dtype=torch.bfloat16, device=ds_engine.device)

        expected_hidden1 = create_tensor([
            [1,
             1,
             1,
             1,
             1],
            [1,
             1,
             1,
             2,
             2],
            [2,
             2,
             2,
             2,
             2],
        ])
        expected_hidden2 = create_tensor([
            [2,
             2,
             2,
             2,
             2],
            [2,
             2,
             2,
             8,
             8],
            [8,
             8,
             8,
             8,
             8],
        ])
        expected_yhat = create_tensor([[6,
                                        6,
                                        6,
                                        6,
                                        6],
                                       [6,
                                        6,
                                        6,
                                        48,
                                        48],
                                       [48,
                                        48,
                                        48,
                                        48,
                                        48]])
        expected_loss = create_tensor([
            [5,
             5,
             5,
             5,
             5],
            [5,
             5,
             5,
             47,
             47],
            [47,
             47,
             47,
             47,
             47],
        ])

        for train_iter in range(3):
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
            activations = ds_engine(
                x=torch.ones((m,
                              n),
                             dtype=torch.bfloat16,
                             device=ds_engine.device),
                y=torch.ones((m,
                              n),
                             dtype=torch.bfloat16,
                             device=ds_engine.device),
                prefetching=train_iter > 0,
            )
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})

            # check the gradients
            grad_partitions = ds_engine.optimizer.get_fp32_grad_partitions()
            assert set(grad_partitions.keys()) == {0}, f"should have one parameter group but got {len(grad_partitions)}"
            assert set(grad_partitions[0].keys()) == {0, 1, 2}
            dloss_wrt_layer1 = grad_partitions[0][0]
            dloss_wrt_layer2 = grad_partitions[0][1]
            dloss_wrt_layer3 = grad_partitions[0][2]

            # layer1 = [..., 1, 2, ...]
            # layer2 = [..., 2, 4, ...]
            # layer3 = [..., 3, 6, ...]
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x

            expected_grad_dtype = torch.float32 if offload_optimizer else torch.bfloat16

            grad_multiplier = 1 if zero_grad else (train_iter + 1)
            if dist.get_rank() == 0:
                assert torch.allclose(
                    dloss_wrt_layer3.cuda(),
                    grad_multiplier * create_tensor([2] * 8).to(expected_grad_dtype))
                assert torch.allclose(
                    dloss_wrt_layer2.cuda(),
                    grad_multiplier * create_tensor([3 * 1] * 8).to(expected_grad_dtype))
                assert torch.allclose(
                    dloss_wrt_layer1.cuda(),
                    grad_multiplier *
                    create_tensor([3 * 2 * 1] * 8).to(expected_grad_dtype))
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(
                    dloss_wrt_layer3.cuda(),
                    grad_multiplier *
                    create_tensor(([8] * 7) + [0]).to(expected_grad_dtype))
                assert torch.allclose(
                    dloss_wrt_layer2.cuda(),
                    grad_multiplier *
                    create_tensor(([6 * 2] * 7) + [0]).to(expected_grad_dtype))
                assert torch.allclose(
                    dloss_wrt_layer1.cuda(),
                    grad_multiplier *
                    create_tensor(([6 * 4 * 1] * 7) + [0]).to(expected_grad_dtype))
            else:
                raise RuntimeError("test has world size of two")

            if zero_grad:
                ds_engine.optimizer.zero_grad()

        # TODO. add testing for this - for now we just call it to make sure it
        # doesn't throw
        ds_engine.optimizer.step()
        _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})

    _test_zero3_param_partitioning()
