from typing import Literal

from torch import cuda
from .abstract import *


__all__ = 'TritonHarness',


@settings_class
class Settings:
    run_time_measured_millis: int = 100
    run_time_warmup_millis: int = 10
    returned_stat: Literal['min', 'max', 'mean', 'median'] = 'mean'


class TritonHarness(BenchmarkHarness[Settings]):
    Settings = Settings

    @classmethod
    def is_available(cls) -> bool:
        if not cuda.is_available():
            return False
        try:
            from triton.testing import do_bench  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def default_settings(cls):
        return Settings()

    @classmethod
    def _get_harness_name(cls):
        return 'triton'

    def measure_runtime_milliseconds(self, fn: BenchMarkedFunction) -> float:
        from triton.testing import do_bench
        s = self.settings
        return do_bench(
            fn, warmup=s.run_time_warmup_millis, rep=s.run_time_measured_millis, return_mode=s.returned_stat)
