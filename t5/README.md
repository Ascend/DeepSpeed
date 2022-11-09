## T5 模型上 DeepSpeed 特性的应用

### 简介
此目录中主要是对 DeepSpeed 特性进行组合，通过 T5 模型去承载验证。
特性包括 FP16、Pipeline parallelism、MoE、ZeRO stage 1/2、ZeRO Offload 等特性组合。
模型使用的是 HuggingFace 上的 T5-small，数据集同样为 HuggingFace datasets 库提供的 wiki_dpr。

### 安装依赖库

请确保环境上有以下依赖库：

```shell
pip install sentencepiece fire loguru sh tensorboard pytz Pillow pytest faiss-cpu decorator sympy datasets==2.2.1 transformers==4.18.0
```

### DeepSpeed 库及 DeepSpeed NPU 插件

请确保已安装 DeepSpeed 库，以及 DeepSpeed NPU 插件。
```shell
pip install deepspeed==0.6.0
git clone https://gitee.com/ascend/DeepSpeed.git --branch adaptor
cd DeepSpeed
pip install ./
```

### 数据集的下载验证

就像之前提过的，模型使用了 HuggingFace 上的 wiki_dpr 数据集作为预训练数据集，模型代码已经集成对数据集的下载。
在第一次运行时请将运行脚本里的 `HF_DATASETS_OFFLINE` 环境变量设置为 0，关闭离线模式，设置 `--dataset_dir` 数据集文件夹参数，下载验证数据集。
在数据集完成下载后，可将 `HF_DATASETS_OFFLINE` 设置回 1，打开离线模式，不再对数据集做验证操作。
注意，此数据集的数据量较大，请准备至少 200 GiB 的硬盘空间。

### 运行模型

模型通过运行 shell 脚本启动，如想运行 T5（FP16 + Pipeline + ZeRO stage 1）模型请使用 `train_t5_pipeline_full_8p.sh` 脚本。
如想运行 T5（FP16 + MoE + ZeRO stage 2）模型请使用 `train_t5_moe_full_8p.sh` 脚本。请将数据集地址作为 data_path 参数传入。

#### T5（FP16 + Pipeline + ZeRO stage 1）

```commandline
bash train_t5_pipeline_full_8p.sh --data_path=/home/datasets/T5
```

#### T5（FP16 + MoE + ZeRO stage 2）

```commandline
bash train_t5_moe_full_8p.sh --data_path=/home/datasets/T5
```

#### T5（FP16 + ZeRO stage 1 + ZeRO Offload）

```commandline
bash train_t5_zero1_offload_full_8p.sh --data_path=/home/datasets/T5
```

#### T5（FP16 + ZeRO stage 2 + ZeRO Offload）

```commandline
bash train_t5_zero2_offload_full_8p.sh --data_path=/home/datasets/T5
```

以上模型相关日志会保存在 `./output/<模型名称>` 文件夹下。

#### T5 ZeRO Offload 内存对比

此脚本会使用 3 亿 6 千万参数量的 T5 模型，运行两次，一次开启 ZeRO Offload，一次不开启，在运行一个 step 后，
通过 `torch.npu.memory_summary()` 函数获取当前时刻下的内存使用量，最后打印两次获取到的内存数值。
（注意，因为 PyTorch 内存特殊申请机制，此方式获取到的内存值只能作为大概参考，并不是准确数值）

```commandline
bash run_t5_zero1_memory.sh --data_path=/home/datasets/T5
```

### 超参介绍

| 参数                      | 解释                  |
|-------------------------|---------------------|
| --include               | 设置计算卡               |
| --checkpoint_dir        | 模型存档文件夹             |
| --tokenizer_name_or_dir | 分词器名称或文件夹           |
| --dataset_dir           | 数据集文件夹              |
| --num_iterations        | 训练步数                |
| --fp16                  | 混合精度模式              |
| --initial_scale_power   | 动态 Loss Scale 的幂初始值 |
| --batch_size            | Batch 大小            |
| --dropout               | Dropout 概率          |
| --num_stage             | Pipeline 并行切分数      |
| --zero_stage            | ZeRO 特性的 stage      |
| --cpu_adam              | 开启 CPU Adam 特性      |
| --print_memory          | 执行一次 step 后打印内存使用量  |
| --moe                   | 开启 MoE 特性           |
| --moe_num_experts       | MoE 专家数             |
| --moe_ep_size           | MoE 专家并行数           |

### 常见环境问题

`libgomp-d22c30c5.so` 无法在静态 TLS 块中分配内存
```
libgomp-d22c30c5.so.1.0.0:cannot allocate memory in static TLS block
```

解决方法：在环境变量`LD_PRELOAD`中添加 `libgomp-d22c30c5.so` 路径
```
export LD_PRELOAD=$LD_PRELOAD:<绝对路径>/libgomp-d22c30c5.so.1.0.0
```
