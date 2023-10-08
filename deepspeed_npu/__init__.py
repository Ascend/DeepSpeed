import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

global FLAG_SUPPORT_INF_NAN
FLAG_SUPPORT_INF_NAN = hasattr(torch_npu.npu.utils, 'is_support_inf_nan') and torch_npu.npu.utils.is_support_inf_nan()

from . import adaptor_utils
from . import adaptor_launcher_runner
from . import adaptor_moe_shared_moe
from . import adaptor_ops_adam_fused_adam
from . import adaptor_ops_op_builder_cpu_adam
from . import adaptor_ops_op_builder_builder
from . import adaptor_runtime_pipe_engine
from . import adaptor_runtime_pipe_module
from . import adaptor_runtime_activation_checkpointing_checkpointing
from . import adaptor_runtime_comm_coalesced_collectives
from . import adaptor_runtime_fp16_fused_optimizer
from . import adaptor_runtime_fp16_loss_scaler
from . import adaptor_runtime_fp16_onebit_adam
from . import adaptor_runtime_fp16_unfused_optimizer
from . import adaptor_runtime_engine
from . import adaptor_runtime_utils
from . import adaptor_zero_partition_parameters
from . import adaptor_zero_stage_1_and_2
from . import adaptor_zero_stage3
from . import adaptor_utils_groups
