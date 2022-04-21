"""unit tests for coalesced collectives"""

import pytest

import torch
import torch.distributed as dist
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced

from .common import distributed_test


@distributed_test(world_size=2)
def test_reduce_scatter_coalesced_single_input():
    input = torch.full((6,
                        ),
                       dist.get_rank(),
                       dtype=torch.half,
                       device=torch.cuda.current_device())

    (output, ) = reduce_scatter_coalesced([input], dist.group.WORLD)

    assert output.shape == (3, )
    assert torch.allclose(output, torch.full_like(output, 0.5))


@distributed_test(world_size=2)
def test_reduce_scatter_coalesced_two_inputs():
    tensor_kwargs = {"device": torch.cuda.current_device(), "dtype": torch.half}
    inputs = [
        dist.get_rank() * torch.arange(0,
                                       6,
                                       **tensor_kwargs),
        dist.get_rank() * torch.arange(6,
                                       9,
                                       **tensor_kwargs),
    ]

    output1, output2 = reduce_scatter_coalesced(inputs, dist.group.WORLD)

    if dist.get_rank() == 0:
        assert output1.shape == (3, )
        assert torch.allclose(output1, torch.arange(0, 3, **tensor_kwargs) / 2)
        assert output2.shape == (2, )
        assert torch.allclose(output2, torch.arange(6, 8, **tensor_kwargs) / 2)
    elif dist.get_rank() == 1:
        assert output1.shape == (3, )
        assert torch.allclose(output1, torch.arange(3, 6, **tensor_kwargs) / 2)
        assert output2.shape == (1, )
        assert torch.allclose(output2, torch.arange(8, 9, **tensor_kwargs) / 2)


@distributed_test(world_size=2)
def test_reduce_scatter_coalesced_tensor_smaller_than_world_sz():
    input = torch.zeros((1, ), dtype=torch.half, device=torch.cuda.current_device())

    (output, ) = reduce_scatter_coalesced([input], dist.group.WORLD)

    if dist.get_rank() == 0:
        assert output.shape == (1, )
        assert torch.allclose(output, torch.zeros_like(output))
    elif dist.get_rank() == 1:
        assert output.shape == (0, )
