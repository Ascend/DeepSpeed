#!/bin/bash
source env_npu.sh
export HF_DATASETS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=240

python run.py --include localhost:0,1,2,3,4,5,6,7 t5_pipeline.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir /home/dataset/T5 \
--fp16 --initial_scale_power 15 --batch_size 32 --dropout 0.1 \
--num_stage 2 --zero_stage 1 \
--num_iterations 100000
