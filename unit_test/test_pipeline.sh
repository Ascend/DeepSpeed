source env_npu.sh
mkdir -p tmp
pytest test_pipe.py ./tmp
pytest test_pipe_module.py
pytest test_pipe_schedule.py
