# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Type

from .abstract import BenchmarkHarness
from .cpu import CPUHarness
from .triton import TritonHarness

_REGISTRY: Dict[str, Type[BenchmarkHarness]] = {
    "cpu": CPUHarness,
    "triton": TritonHarness,
}


def get_harness(name: Optional[str] = None) -> BenchmarkHarness:
    """Return a benchmark harness by name, or auto-select the best available."""
    if name is not None:
        if name not in _REGISTRY:
            raise ValueError(f"Unknown harness: {name!r}. Available: {list(_REGISTRY.keys())}")
        harness = _REGISTRY[name]()
        if not harness.is_available():
            raise RuntimeError(f"Harness {name!r} is not available in the current environment")
        return harness

    # Auto-select: prefer triton, fall back to cpu
    for candidate_name in ("triton", "cpu"):
        harness = _REGISTRY[candidate_name]()
        if harness.is_available():
            return harness

    raise RuntimeError("No benchmark harness available")
