from deepspeed.accelerator import get_accelerator
from deepspeed.utils import timer


def get_elapsed_msec(self):
    self.end_event.synchronize()
    return self.start_event.elapsed_time(self.end_event)


timer.CudaEventTimer.get_elapsed_msec = get_elapsed_msec