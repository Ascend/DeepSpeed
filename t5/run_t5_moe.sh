#!/bin/bash
source env_npu.sh
export HF_DATASETS_OFFLINE=1


python run.py --include localhost:0,1,2,3,4,5,6,7 t5_moe.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir /home/dataset/T5 \
--fp16 --initial_scale_power 15 --batch_size 8 --dropout 0.1 \
--zero_stage 2 --moe --moe_num_experts 16 --moe_ep_size 4 \
--num_iterations 50000
