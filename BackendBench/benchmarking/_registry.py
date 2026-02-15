from typing import Mapping

from .abstract import BenchmarkHarness
from .cpu import CPUHarness
from .triton import TritonHarness
from .cuda import CUDAHarness

harness_class_by_name: Mapping[str, type[BenchmarkHarness]] = {
    cls.harness_name: cls for cls in (
        CPUHarness, TritonHarness, CUDAHarness,
    )
}
