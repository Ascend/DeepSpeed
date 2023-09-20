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

1. 先安装原生 deepspeed

```bash
pip3 install deepspeed==0.9.2
```

2. 然后安装 deepspeed-npu 插件

```bash
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install .
```

### 3.插件使用方法

在入口文件行首 import deepspeed_npu，并配合 deepspeed / torch 使用,例如

```python
import deepspeed_npu
import torch
import torch_npu
...
```

### 3. DeepSpeed 使用参考
[https://github.com/microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)