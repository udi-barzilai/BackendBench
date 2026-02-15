# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable


class BenchmarkHarness(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def bench(self, fn: Callable[[], Any]) -> float:
        """Benchmark fn and return execution time in milliseconds."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this harness can run in the current environment."""
        ...
