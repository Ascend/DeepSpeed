## deepspeed_npu:Ascend NPU适配deepspeed插件

通过deepspeed_npu，你可以在Ascend910芯片上使用deepspeed，并基于deepspeed进行开发。目前，deepspeed_npu主要支持以下特性

1. fp16
2. Pipeline Parallelism
3. ZeRO（stage1-stage3）
4. ZeRO-Offload
5. One-bit Adam
6. MoE
7. Activation checkpointing

请参考deepspeed官方文档获取这些特性的详细说明：https://www.deepspeed.ai/

### 1.版本说明

目前仅支持deepspeed版本0.6.0：https://github.com/microsoft/DeepSpeed/tree/v0.6.0

### 2.安装方法

1.先安装原生deepspeed

```
pip3 install deepspeed==0.6.0
```

2.然后安装deepspeed-npu插件

```
git clone https://gitee.com/ascend/DeepSpeed.git
cd DeepSpeed
pip3 install ./
```

***注：deepspeed-npu安装后会修改命令行deepspeed和ds，请按照顺序安装，并使用匹配的版本。***

### 3.插件使用方法

在入口文件行首import deepspeed_npu，并配合deepspeed/torch使用,例如

```
import deepspeed_npu
import torch
from deepspeed import distributed_test
...
```

### 4.运行单元测试

进入unit_test目录，运行各特性的单元测试

fp16:

```
bash test_fp16.sh
```

Pipeline:

```
bash test_pipeline.sh
```

ZeRO:

```
bash test_zero.sh
```

one-bit Adam:

```
bash test_onebit_adam.sh
```

MoE:

```
bash test_moe.sh
```

### 5.运行T5模型使用

请参考提供的T5模型[使用指导](./t5/README.md)