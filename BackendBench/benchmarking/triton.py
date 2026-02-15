# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch

from .abstract import BenchmarkHarness


class TritonHarness(BenchmarkHarness):
    def __init__(self):
        super().__init__("triton")

    def is_available(self) -> bool:
        try:
            if torch.cuda.is_available():
                import triton.testing  # noqa: F401

                return True
            return False
        except ImportError:
            return False

    def bench(self, fn: Callable[[], Any]) -> float:
        import triton.testing

        return triton.testing.do_bench(fn)
