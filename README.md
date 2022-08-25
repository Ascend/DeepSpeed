## Ascend NPU适配deepspeed插件

### 1.版本说明

基于deepspeed版本0.6.0：https://github.com/microsoft/DeepSpeed/tree/v0.6.0

### 2.安装方法

首先安装原生deepspeed

```
pip3 install deepspeed==0.6.0
```

然后在工程根目录安装插件

```
pip3 install ./
```

### 3.使用方法

在入口文件首import deepspeed_npu，并配合deepspeed/torch使用,例如

```
import deepspeed_npu
import torch
import deepspeed
```

