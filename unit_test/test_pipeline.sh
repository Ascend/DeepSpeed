source env_npu.sh
mkdir -p tmp
pytest -s test_pipe.py ./tmp
pytest -s test_pipe_module.py
pytest -s test_pipe_schedule.py
