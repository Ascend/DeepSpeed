#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="T5_MoE"
# 训练batch_size
batch_size=8
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""


train_iterations=1500


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

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
test_path_dir=${cur_path}



#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${Network} ];then
    rm -rf ${test_path_dir}/output/${Network}
    mkdir -p ${test_path_dir}/output/$Network
else
    mkdir -p ${test_path_dir}/output/$Network
fi


#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source env_npu.sh
fi

# HuggingFace 数据集离线模式，1开启、0关闭（离线数据集下载完第一次需要验证完整性）
export HF_DATASETS_OFFLINE=1

echo "Start training..."

nohup python run.py --include localhost:0,1,2,3,4,5,6,7 t5_model.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir ${data_path} \
--fp16 --initial_scale_power 15 --batch_size ${batch_size} --dropout 0.1 \
--zero_stage 2 --moe --moe_num_experts 16 --moe_ep_size 4 \
--num_iterations ${train_iterations} > ${test_path_dir}/output/${Network}/train_${Network}.log 2>&1 &

wait


##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "Duration" ${test_path_dir}/output/${Network}/train_${Network}.log | tail +2 | awk '{print $13}' >> ${test_path_dir}/output/${Network}/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/${Network}/train_${CaseName}_fps.log | awk -F: '{print ($1 * 3600) + ($2 * 60) + $3}'| awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# 打印，不需要修改
echo "Step average duration: $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'finished training, last'  ${test_path_dir}/output/${Network}/train_${Network}.log | awk '{print $16}'`
# 打印，不需要修改
echo "Last 1K average loss: ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
echo "Total iterations: ${train_iterations}"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`


# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/${Network}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log
echo "TrainingIterations = ${train_iterations}" >>  ${test_path_dir}/output/${Network}/${CaseName}.log