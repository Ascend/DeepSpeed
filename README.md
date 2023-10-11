## deepspeed_npu: Ascend NPU 适配 deepspeed 插件

通过 deepspeed_npu，你可以在 Ascend910 芯片上使用 deepspeed，并基于 deepspeed 进行开发。目前，deepspeed_npu 主要支持以下特性:

1. FP16
2. Gradient Accumulation
3. Data Parallelism
4. Pipeline Parallelism
5. Tensor Parallelism (Inference Engine)
6. ZeRO (stage1-stage3)
7. Activation Checkpointing
8. ZeRO-Offload
9. CPU Adam
10. Fused Adam
11. One-bit Adam
12. MoE
13. Zero Infinity
14. Zero-One Adam
15. Curriculum Learning
16. Progressive layer dropping

请参考 deepspeed 官方文档获取这些特性的详细说明：https://www.deepspeed.ai/

### 1.版本说明

目前仅支持 deepspeed 版本 0.9.2：https://github.com/microsoft/DeepSpeed/tree/v0.9.2

### 2.安装方法
 
#### 2.1 先安装原生 deepspeed

```bash
pip3 install deepspeed==0.9.2
```

#### 2.2 然后安装 deepspeed-npu 插件

```bash
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install .
```

#### 2.3 安全加固（可选）

##### 2.3.1 权限相关说明

  - 运行程序前，建议用户对训练所需文件做好权限控制等安全措施，请勿使用管理员账户安装运行，文件夹权限建议设置为 750，文件权限建议设置为 640。
  - 在多用户共享数据集的场景下，请根据需求最小化权限设置所需的文件夹以及文件的读写权限等，避免出现非法访问等安全问题。
  - 对于涉及隐私数据、商业资产等敏感文件，建议用户做好安全防护和权限控制，避免隐私泄露造成安全风险。
  - 对于涉及到使用 C++ 动态编译特性的场景，建议打开 ASLR （地址空间配置随机加载）以及对编译后的 SO 文件开启 strip（移除调试符号信息），减少程序的暴露面。 因编译由 DeepSpeed 原生框架负责且无此类配置选项，故需用户自行开启，开启方法参考下方章节。

##### 2.3.2 打开 ASLR
```shell
echo 2 > /proc/sys/kernel/randomize_va_space
```

##### 2.3.3 对 cpu_adam、offload 等特性动态编译的 so 文件开启 strip
```shell
strip -s /PATH/FEATURE.so
```


### 3.插件使用方法

在入口文件行首 import deepspeed_npu，并配合 deepspeed / torch 使用,例如

```python
import deepspeed_npu
import torch
import torch_npu

...
```

### 4. DeepSpeed 使用参考

[https://github.com/microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)

### 5. 关于

#### 5.1 目录结构说明

- `deepspeed_npu`：文件夹下的各个文件都对应原生的文件，如 adaptor_xxx_yyy.py 文件对应原生的 xxx.yyy.py 文件。
- `deepspeed_npu.csrc_npu`：文件夹下为相关特性的动态编译 C++ 文件，与原生的 csrc 文件夹相对应。如 csrc_npu/adam 文件夹对应原生的 csrc/adam 文件夹。

#### 5.2 通信矩阵说明

本插件不涉及端口侦听等相关行为，相关端口由用户在模型脚本指定调用原生接口开启，建议用户注意做好安全防护，单机训练的情况下请勿绑定全局端口。

#### 5.3 接口替换说明

deepspeed_npu 以 **monkey patching/装饰器**等方式**替换/修改** DeepSpeed 原有函数实现，并不提供对外接口，用户只需要`import deepspeed_npu`，做到无感迁移原有模型代码。

以下是 patch 的相关接口与变量：

| 接口与变量                                                                                                      | 简介                              |
|------------------------------------------------------------------------------------------------------------|---------------------------------|
| deepspeed.launcher.EXPORT_ENVS                                                                             | 多机环境变量                          |
| deepspeed.moe._ALLToALL                                                                                    | all to all 通信函数                 |
| deepspeed.moe.sharded_moe.top1gating                                                                       | moe top1 获取函数                   |
| deepspeed.ops.adam.FusedAdam.__init__                                                                      | 融合优化器初始化函数                      |
| deepspeed.ops.adam.FusedAdam.step                                                                          | 融合优化器权重更新函数                     |
| deepspeed.ops.op_builder.async_io.AsyncIOBuilder.sources                                                   | AsyncIOBuilder 源文件路径获取函数        |
| deepspeed.ops.op_builder.async_io.AsyncIOBuilder.include_paths                                             | AsyncIOBuilder include 文件路径获取函数 |
| deepspeed.ops.op_builder.builder.assert_no_cuda_mismatch                                                   | cuda 版本检查函数                     |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.torch_npu_path                                            | CANN 路径获取函数                     |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.ascend_path                                               | torch npu 路径获取函数                |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.sources                                                   | CPUAdamBuilder 源文件路径获取函数        |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.include_paths                                             | CPUAdamBuilder include 文件路径获取函数 |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.extra_ldflags                                             | ld 选项获取函数                       |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.cxx_args                                                  | c++ 编译选项获取函数                    |
| deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.nvcc_args                                                 | cuda 编译选项获取函数                   |
| deepspeed.runtime.activation_checkpointing.checkpointing.CheckpointFunction.backward                       | 从计算反向函数                         |
| deepspeed.runtime.comm.coalesced_collectives.torch_reduce_scatter_fn                                       | reduce scatter 装饰器              |
| deepspeed.runtime.engine.DeepSpeedEngine._configure_basic_optimizer                                        | 优化器设置函数                         |
| deepspeed.runtime.fp16.fused_optimizer.FP16_Optimizer.step                                                 | 优化器权重更新函数                       |
| deepspeed.runtime.fp16.fused_optimizer.FP16_Optimizer.backward                                             | 优化器反向函数                         |
| deepspeed.runtime.fp16.loss_scaler.LossScalerBase.backward                                                 | LossScaler 反向函数                 |
| deepspeed.runtime.fp16.loss_scaler.DynamicLossScaler.has_overflow_serial                                   | 溢出检测函数                          |
| deepspeed.runtime.fp16.onebit.adam.OnebitAdam.__init__                                                     | OnebitAdam 初始化函数                |
| deepspeed.runtime.fp16.onebit.zoadam.ZeroOneAdam.__init__                                                  | ZeroOneAdam 初始化函数               |
| deepspeed.runtime.fp16.unfused_optimizer.FP16_UnfusedOptimizer.step                                        | 优化器权重更新函数                       | 
| deepspeed.runtime.fp16.unfused_optimizer.FP16_UnfusedOptimizer.backward                                    | 优化器反向函数                         | 
| deepspeed.runtime.pipe.engine.PipelineEngine.ID_TO_DTYPE                                                   | ID转Dtype列表                      |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine.DTYPE_TO_ID                                            | Dtype转ID字典                      |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._INSTRUCTION_MAP[schedule.BackwardPass]                | pipeline engine 反向指令            |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._INSTRUCTION_MAP[schedule.RecvActivation]              | pipeline engine 接收激活值指令         |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._INSTRUCTION_MAP[schedule.RecvGrad]                    | pipeline engine 接收梯度指令          |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._INSTRUCTION_MAP[schedule.SendGrad]                    | pipeline engine 发送梯度指令          |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._INSTRUCTION_MAP[schedule.ReduceGrads]                 | pipeline engine 同步梯度指令          |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._exec_backward_pass                                    | pipeline engine 反向函数            |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._send_tensor_meta                                      | 发送 tensor meta 信息函数             |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._recv_tensor_meta                                      | 接受 tensor meta 信息函数             |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._exec_recv_activations                                 | 接收激活值函数                         |                          
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._exec_recv_grads                                       | 接收梯度函数                          |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._exec_send_grads                                       | 发送梯度函数                          |
| deepspeed.runtime.pipe.engine.engine.PipelineEngine._exec_reduce_grads                                     | 同步梯度函数                          |
| deepspeed.runtime.pipe.module.PipelineModule._partition_layers                                             | layer 切分函数                      |
| deepspeed.runtime.pipe.module.PipelineModule._is_checkpointable                                            | 重计算判断函数                         |
| deepspeed.runtime.utils.CheckOverflow.check_using_norm                                                     | 溢出同步函数                          |
| deepspeed.runtime.utils.CheckOverflow.has_overflow_serial                                                  | 溢出检测函数                          |
| deepspeed.runtime.utils.CheckOverflow.has_overflow                                                         | 溢出检测函数                          |
| deepspeed.runtime.utils.CheckOverflow.get_grad_norm                                                        | 梯度归一化函数                         |
| deepspeed.runtime.utils.CheckOverflow.get_weight_norm                                                      | 权杖归一化函数                         |
| deepspeed.comm.dist.send                                                                                   | 发送函数                            |
| deepspeed.comm.dist.recv                                                                                   | 接收函数                            |
| deepspeed.comm.dist.all_reduce                                                                             | all reduce 函数                   |
| deepspeed.utils.timer.SynchronizedWallClockTimer.Timer.__init__                                            | Timer 初始化函数                     |
| deepspeed.utils.timer.SynchronizedWallClockTimer.Timer.start                                               | Timer 开始函数                      |
| deepspeed.utils.timer.SynchronizedWallClockTimer.Timer.stop                                                | Timer 停止函数                      |
| deepspeed.utils.timer.SynchronizedWallClockTimer.Timer._get_elapsed_msec                                   | Timer 计时函数                      |
| deepspeed.utils.timer.SynchronizedWallClockTimer.Timer.reset                                               | Timer 重置函数                      |
| deepspeed.runtime.zero.mics.MiCS_Optimizer.allreduce_mics_shard_grads                                      | 共享梯度同步函数                        |
| deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.complete_grad_norm_calculation_for_cpu_offload | offload 梯度归一化函数                 |
| deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.partition_grads                                | 梯度切分函数                          |
| deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow_serial                            | 溢出检测函数                          |
| deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow_partitioned_grads_serial          | 溢出检测函数                          |
| deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.has_overflow                                   | 溢出检测函数                          |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.update_overflow_tracker_for_param_grad         | 溢出检测更新函数                        |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.complete_grad_norm_calculation_for_cpu_offload | offload 梯度归一化函数                 |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.get_grad_norm_direct                           | 梯度归一化函数                         |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow_serial                            | 溢出检测函数                          |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow_partitioned_grads_serial          | 溢出检测函数                          |
| deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.has_overflow                                   | 溢出检测函数                          |
| deepspeed.checkpoint.utils.clone_tensors_for_torch_save                                                    | tensor 保存函数                     |
| torch.nn.functional.one_hot                                                                                | one hot 函数                      |
| torch.jit.script                                                                                           | jit script 函数                   |
| torch.cuda.nvtx                                                                                            | cuda 函数                         |

