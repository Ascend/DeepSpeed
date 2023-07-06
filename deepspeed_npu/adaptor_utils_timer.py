import time
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import timer


def TimerInit(self, name):
    self.name_ = name
    self.elapsed_ = 0.0
    self.started_ = False
    self.start_time = time.time()
    self.records = []


def start(self):
    """Start the timer."""
    assert not self.started_, f"{self.name_} timer has already been started"
    get_accelerator().synchronize()
    self.start_time = time.time()
    self.started_ = True


def stop(self, reset=False, record=False):
    """Stop the timer."""
    assert self.started_, "timer is not started"
    get_accelerator().synchronize()
    if reset:
        self.elapsed_ = time.time() - self.start_time
    else:
        self.elapsed_ += time.time() - self.start_time
    self.started_ = False
    if record:
        self.records.append(self.elapsed_)


def _get_elapsed_msec(self):
    return self.elapsed_


def reset(self):
    """Reset timer."""
    self.elapsed_ = 0.0
    self.started_ = False
    self.acc_ = 0.0
    self.cnt_ = 0


timer.SynchronizedWallClockTimer.Timer.__init__ = TimerInit
timer.SynchronizedWallClockTimer.Timer.start = start
timer.SynchronizedWallClockTimer.Timer.stop = stop
timer.SynchronizedWallClockTimer.Timer._get_elapsed_msec = _get_elapsed_msec
timer.SynchronizedWallClockTimer.Timer.reset = reset