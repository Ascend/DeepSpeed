source env_npu.sh
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_ALGO="level0:fullmesh;level1:fullmesh"

mkdir -p tmp
pytest test_pipe.py ./tmp
pytest test_pipe_module.py
pytest test_pipe_schedule.py
