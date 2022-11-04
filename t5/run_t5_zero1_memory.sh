#!/bin/bash
# 数据集路径,保持为空,不需要修改
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done


# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source env_npu.sh
fi

# 执行模型 without offload
no_offload_memory=`python run.py --include localhost:0 t5_model.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir ${data_path} \
--fp16 --initial_scale_power 15 --batch_size 8 --dropout 0.1 \
--zero_stage 1 --print_memory \
--num_iteration 1 \
--num_heads 8 --ff_dim 2048 --d_model 3072 --max_seq_length 128 | grep "Allocated memory" | awk '{printf $5 $6}'`

# 执行模型 with offload
offload_memory=`python run.py --include localhost:0 t5_model.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir ${data_path} \
--fp16 --initial_scale_power 15 --batch_size 8 --dropout 0.1 \
--zero_stage 1 --cpu_adam --print_memory \
--num_iteration 1 \
--num_heads 8 --ff_dim 2048 --d_model 3072 --max_seq_length 128 | grep "Allocated memory" | awk '{printf $5 $6}'`


echo "------------------ Final result ------------------"
echo "Allocated memory without ZeRO offload: ${no_offload_memory}"
echo "Allocated memory with ZeRO offload: ${offload_memory}"