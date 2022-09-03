source env_npu.sh
pytest test_zero_context.py::test_scattered_init_dist
pytest test_zero_context.py::test_scatter_gather
pytest test_zero_context.py::test_gather_update
pytest test_zero_context.py::test_ext_param_getattr
pytest test_zero_context.py::test_scatter_halftype
pytest test_zero_context.py::test_ext_param_return
pytest test_zero_context.py::test_ext_param_returnobj
pytest test_zero_context.py::test_stage_3_output_type
pytest test_zero_context.py::test_subclass_param
pytest test_zero_context.py::test_subclass_param_init

pytest test_zero_tiled.py
pytest test_zero.py