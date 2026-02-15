from time import perf_counter as get_perf_counter

from ._common import *
from .abstract import *


__all__ = 'CPUHarness',


@settings_class
class Settings:
    run_count_measured: int = 100
    run_count_warmup: int = 10


class CPUHarness(BenchmarkHarness[Settings]):
    Settings = Settings

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def default_settings(cls):
        return Settings()

    @classmethod
    def _get_harness_name(cls) -> str:
        return 'cpu'

    def measure_runtime_milliseconds(self, fn: BenchMarkedFunction) -> float:
        s = self.settings
        for _ in range(s.run_count_warmup):
            fn()
        start = get_perf_counter()
        for _ in range(s.run_count_measured):
            fn()
        elapsed_s = (get_perf_counter() - start) / s.run_count_measured
        return elapsed_s * millis_per_second
