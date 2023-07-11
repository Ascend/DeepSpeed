"""Entrypoint for deepspeed_npu that imports deepspeed_npu module before running the deepspeed runner"""

from deepspeed.launcher.runner import main

def deepspeed_npu_main():
    import deepspeed_npu # noqa
    main()