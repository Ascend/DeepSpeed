import deepspeed


def assert_no_cuda_mismatch():
    pass


deepspeed.ops.op_builder.builder.assert_no_cuda_mismatch = assert_no_cuda_mismatch
