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

  - 请勿使用管理员账户安装运行，建议安装完成后对安装目录文件做好权限管控，文件夹权限建议设置为 750，文件权限建议设置为 640。
  - 如需要保存安装日志，可在安装命令后面加上参数`--log <PATH>`， 对`<PATH>` 做好权限管控。
  - 运行程序前，建议用户对训练所需文件，模型脚本、权重等做好权限控制等安全措施。
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
