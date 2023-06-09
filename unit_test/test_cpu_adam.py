import argparse
import torch
import torch_npu
import time
import numpy as np
import pytest
import copy

import deepspeed
import deepspeed_npu
from deepspeed.ops.op_builder import CPUAdamBuilder

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible")


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)

@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             (55),
                             (127),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
def test_cpu_adam_opt(model_size):
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    device = 'cpu'
    torch.npu.set_device(0)
    rng_state = torch.get_rng_state()
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)
    param1 = torch.nn.Parameter(torch.randn(model_size, device=device))

    optimizer1 = torch.optim.AdamW([param1])
    optimizer = DeepSpeedCPUAdam([param])

    for i in range(10):
        rng_state = torch.get_rng_state()
        param.grad = torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        param1.grad = torch.randn(model_size, device=device)

        optimizer.step()
        optimizer1.step()

    check_equal(param, param1, atol=1e-2, verbose=True)


def test_cpu_adam_npu_error():
    model_size = 64
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    device = 'npu:0'
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    optimizer = DeepSpeedCPUAdam([param])

    param.grad = torch.randn(model_size, device=device)
    with pytest.raises(AssertionError):
        optimizer.step()
