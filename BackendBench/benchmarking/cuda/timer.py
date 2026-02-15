from dataclasses import dataclass
from typing import Callable, Any, NamedTuple
# noinspection PyPep8Naming
from torch import Tensor, device as Device, empty, uint8, cuda
from torch.cuda import Event, Stream, get_device_properties, synchronize as device_synchronize, current_stream


__all__ = 'CUDAWorkTimer',


class CUDAWorkTimer:
    @classmethod
    def is_available(cls):
        return cuda.is_available() and callable(_get_cuda_delay_function())

    @dataclass(kw_only=True)
    class Settings:
        enable_l2_flush: bool = True
        enable_device_synchronize_before: bool = True
        enable_device_synchronize_after: bool = True
        cycle_count_pre_launch_delay: int = 50 * 1_500_000   # ~50ms with a 1500MHz clock

    def __init__(self, device: Device, *, settings: Settings = Settings()):
        self.settings = settings
        # Use a separate stream for timed kernels only (aids visibility when profiling at system level)
        self.compute_stream = cs = Stream(device=device)
        self.timing_events = te = EventPair.create(enable_timing=True)
        pre_launch_sync_event = Event(enable_timing=False)

        # "warmup" record events to force creation if lazy
        with cs:
            for e in te + (pre_launch_sync_event,):
                e.record(cs)

        def pre_launch():
            s = self.settings  # reload up-to-date settings
            d = cs.device
            if s.enable_device_synchronize_before:
                device_synchronize(d)
            if s.enable_l2_flush:
                _flush_l2_cache(device=d)
            with current_stream(device=d) as pre_launch_stream:
                if (cycle_count := s.cycle_count_pre_launch_delay) > 0:
                    if callable(cuda_delay := _get_cuda_delay_function()):
                        cuda_delay(cycle_count)
                    else:
                        raise RuntimeError(
                            f"{type(self).__name__!r} cannot be used pre-launch delay because the CUDA delay kernel"
                            " failed to compile and/or load.")
                pre_launch_sync_event.record(pre_launch_stream)  # on pre-launch stream
            cs.wait_event(pre_launch_sync_event)
            cs.__enter__()
            te.start.record(cs)
        self._pre_launch = pre_launch

        def post_launch():
            te.stop.record(cs)
            cs.__exit__(*(None,)*3)
            if self.settings.enable_device_synchronize_after:
                device_synchronize(cs.device)
        self._post_launch = post_launch

    __slots__ = 'settings', 'timing_events', 'compute_stream', '_pre_launch', '_post_launch'

    def __enter__(self):
        self._pre_launch()
        return self

    def __exit__(self, *_):
        self._post_launch()


class EventPair(NamedTuple):
    start: Event
    stop: Event

    @classmethod
    def create(cls, *, enable_timing: bool):
        return cls(Event(enable_timing=enable_timing), Event(enable_timing=enable_timing))
    def __enter__(self):
        self.start.record()
    def __exit__(self, *_):
        self.stop.record()
    def synchronize(self):
        self.start.synchronize()
        self.stop.synchronize()
    def elapsed_millis(self):
        self.synchronize()
        return self.start.elapsed_time(self.stop)


class L2Flusher:
    """Caches per-device buffers to be used for flushing device L2 cache"""
    class _Buffers(dict[Device, Tensor]):
        def __missing__(self, device: Device):
            dp = get_device_properties(device=device)
            self[device] = buffer = empty(size=(dp.L2_cache_size, 2), dtype=uint8, device=device)
            return buffer

    def __init__(self):
        self._buffers = self._Buffers()

    def __call__(self, device: Device):
        self._buffers[device].zero_()


_flush_l2_cache = L2Flusher()


def _get_cuda_delay_function():
    func: Callable[[int], None]|None|bool = _get_cuda_delay_function.cached_value
    if func is None:
        # import here to avoid making this a package init-time dependency
        # noinspection PyBroadException
        try:
            from ._delay_kernel import cuda_busy_wait as func
        except Exception:
            func = False
        _get_cuda_delay_function.cached_value = func
    return func
_get_cuda_delay_function.cached_value = None


