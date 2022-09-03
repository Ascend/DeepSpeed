source env_npu.sh
pytest -s test_zero_tiled.py
pytest -s test_zero_context.py
pytest -s test_zero.py