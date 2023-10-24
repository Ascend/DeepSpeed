import torch
import torch.distributed as dist
import numpy as np
from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb


def OnebitLambInit(self,
                   params,
                   deepspeed=None,
                   lr=1e-3,
                   freeze_step=100000,
                   bias_correction=True,
                   betas=(0.9, 0.999),
                   eps=1e-8,
                   eps_inside_sqrt=False,
                   weight_decay=0.,
                   max_grad_norm=0.,
                   max_coeff=10.0,
                   min_coeff=0.01,
                   amsgrad=False,
                   cuda_aware=False,
                   comm_backend_name='nccl',
                   coeff_beta=0.9,
                   factor_max=4.0,
                   factor_min=0.5,
                   factor_threshold=0.1):
    if amsgrad:
        raise RuntimeError('1-bit Lamb does not support the AMSGrad variant.')

    defaults = dict(lr=lr,
                    bias_correction=bias_correction,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    max_coeff=max_coeff,
                    min_coeff=min_coeff)

    super(OnebitLamb, self).__init__(params, defaults)
    self.eps_mode = 0 if eps_inside_sqrt else 1
    assert (dist.is_initialized())

    self.deepspeed = deepspeed
    self.lamb_freeze_key = False
    self.initialize = False
    self.freeze_step = freeze_step
    self.cuda_aware = cuda_aware
    self.coeff_beta = coeff_beta
    self.factor_max = factor_max
    self.factor_min = factor_min
    self.factor_threshold = factor_threshold
    self.using_pipeline = False

    self.comm_backend_name = comm_backend_name

    # Empty initializer. Set handle based on the comm backend as follows.
    self.comm_backend_handle = None

    if self.comm_backend_name == 'nccl':
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        assert (
                (TORCH_MAJOR == 1 and TORCH_MINOR >= 8) or TORCH_MAJOR >= 2
        ), "Please use torch 1.8 or greater to enable NCCL backend in 1-bit Adam. Alternatively, " \
           "please specify 'mpi' as the 'comm_backend_name' in config file to proceed with the MPI backend"
        assert dist.is_initialized() == True, "Please initialize the torch distributed backend."
        from deepspeed.runtime.comm.nccl import NcclBackend
        self.using_pipeline = hasattr(self.deepspeed,
                                      'pipeline_enable_backward_allreduce')
        self.comm_backend_handle = NcclBackend(self.deepspeed.mpu)

    elif self.comm_backend_name == 'mpi':
        from deepspeed.runtime.comm.mpi import MpiBackend
        self.comm_backend_handle = MpiBackend(cuda_aware)

    elif self.comm_backend_name == 'hccl':
        assert dist.is_initialized(), "Please initialize the torch distributed backend."
        from deepspeed_npu.adaptor_runtime_comm_hccl import HcclBackend
        self.using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
        self.comm_backend_handle = HcclBackend(self.deepspeed.mpu)

    self.size = self.comm_backend_handle.size

    self.divider = int(self.size * 8 / np.gcd(self.size, 8))

    self.exp_avg_flat = []
    self.dummy_exp_avg = {}
    self.corrected_tensor_sizes = []
    self.server_chunk_sizes = []
    self.worker_errors = []
    self.server_errors = []

    self.lamb_coeffs = []


OnebitLamb.__init__ = OnebitLambInit
