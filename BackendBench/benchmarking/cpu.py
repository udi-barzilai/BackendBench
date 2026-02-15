# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Callable

from .abstract import BenchmarkHarness


class CPUHarness(BenchmarkHarness):
    def __init__(self):
        super().__init__("cpu")

    def is_available(self) -> bool:
        return True

    def bench(self, fn: Callable[[], Any], num_runs: int = 100) -> float:
        for _ in range(10):
            fn()

        start = time.perf_counter()
        for _ in range(num_runs):
            fn()
        elapsed_s = (time.perf_counter() - start) / num_runs
        return elapsed_s * 1000.0
