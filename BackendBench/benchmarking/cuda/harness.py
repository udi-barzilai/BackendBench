
from ..abstract import *
from .timer import CUDAWorkTimer as Timer


__all__ = 'CUDAHarness',


@settings_class
class Settings:
    device_name: str = None  # None = use the current CUDA device at harness construction time
    run_count_measured: int = 100
    run_count_warmup: int = 10
    outlier_iqr_threshold: float = 1.5
    cuda: Timer.Settings = Timer.Settings()


class CUDAHarness(BenchmarkHarness[Settings]):
    """
    Benchmaking harness that measures the runtime of kernels posted by the `BenchMarkedFunction`
    on the "current stream" of the CUDA device specified at construction time.
    IMPORTANT: Only kernels (and sync waits) posted on that stream, on that device will be measured.
    It will NOT measure any CUDA work posted on other streams / devices, unless it is synchronized into the current
    stream of the specified device.
    """
    Settings = Settings

    def __init__(self, settings: Settings = None):
        super().__init__(settings)
        settings = self.settings  # defaults if `settings` was None
        from torch import cuda, device as torch_device
        dn = settings.device_name
        self.timer = Timer(
            device=torch_device(type='cuda', index=cuda.current_device()) if dn is None else torch_device(dn),
            settings=settings.cuda
        )

    @classmethod
    def is_available(cls):
        return Timer.is_available()

    @classmethod
    def default_settings(cls):
        return Settings()

    @classmethod
    def _get_harness_name(cls):
        return 'cuda'

    def measure_runtime_milliseconds(self, fn: BenchMarkedFunction) -> float:
        s = self.settings
        timer = self.timer
        from torch import empty, float32 as meas_dtype

        for _ in range(s.run_count_warmup):
            fn()
        run_times = empty(s.run_count_measured, dtype=meas_dtype, device='cpu')
        for i in range(s.run_count_measured):
            with timer:
                fn()
            run_times[i] = timer.timing_events.elapsed_millis()

        oit = s.outlier_iqr_threshold
        if oit is not None:
            # IQR outlier removal (see https://en.wikipedia.org/wiki/Interquartile_range)
            global _iqr_quantiles
            if _iqr_quantiles is None:
                _iqr_quantiles = run_times.new_tensor((0.25, 0.75))
            q1, q3 = run_times.quantile(_iqr_quantiles)
            span = (q3 - q1) * s.outlier_iqr_threshold
            is_not_outlier = run_times > q1 - span
            is_not_outlier.logical_and_(run_times < q3 + span)
            run_times = run_times[is_not_outlier]

        return run_times.mean().item()


_iqr_quantiles = None