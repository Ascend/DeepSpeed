from deepspeed.utils import nvtx
def instrument_w_nvtx(func):
    return func
nvtx.instrument_w_nvtx = instrument_w_nvtx