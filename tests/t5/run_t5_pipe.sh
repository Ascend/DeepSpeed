source env.sh
export HF_DATASETS_OFFLINE=1
export LD_PRELOAD=/usr/local/lib/python3.7/dist-packages/faiss/../faiss_cpu.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD

ds --include localhost:0,1,2,3,4,5,6,7 t5_pipe.py \
--checkpoint_dir ./checkpoint \
--tokenizer_name_or_dir ./tokenizer \
--dataset_dir /home/dataset/T5 \
--num_stage 2 --fp16 --zero_stage 1
