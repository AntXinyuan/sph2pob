import time, functools
import torch

class Timer:
    def __init__(self, name='Timer', sync=False, on_off=True):
        self.clock_time = time.perf_counter
        self.name = name
        self.sync = sync
        self.on_off = on_off

    def tic(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start = self.clock_time()

    def toc(self):
        if self.sync:
            torch.cuda.synchronize()
        self.end = self.clock_time()

    @property
    def duration(self):
        assert self.end > self.start
        return self.end - self.start

    def show(self):
        if self.on_off:
            print('%s runs in %.4fs.' % (self.name, self.duration))

    def __enter__(self):
        if self.on_off:
            self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.on_off:
            self.toc()
            self.show()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.name =='Timer':
                self.name = func.__name__
            with self:
                val = func(*args, **kwargs)
            return val
        return wrapper


print_run_time = Timer()