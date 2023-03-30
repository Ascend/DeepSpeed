from deepspeed.launcher import runner


runner.EXPORT_ENVS = ["NCCL", "PYTHON", "MV2", "UCX", "ASCEND", "HCCL", "LD_LIBRARY", "PATH"]