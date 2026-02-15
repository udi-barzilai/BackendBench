from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic, Optional
from dataclasses import dataclass


__all__ = 'BenchmarkHarness', 'BenchMarkedFunction', 'settings_class'


BenchMarkedFunction = Callable[[], Any]

settings_class = dataclass(kw_only=True)  # <- decorator
_Settings = TypeVar('_Settings')


class BenchmarkHarness(Generic[_Settings], ABC):
    @classmethod
    def create(cls, name: Optional[str] = None) -> 'BenchmarkHarness':
        """Factory: create a harness by name, or auto-select the best available."""
        from ._registry import harness_class_by_name, auto_select_preference

        if name is not None:
            if name not in harness_class_by_name:
                raise ValueError(
                    f"Unknown harness: {name!r}. Available: {list(harness_class_by_name)}")
            harness_cls = harness_class_by_name[name]
            if not harness_cls.is_available():
                raise RuntimeError(f"Harness {name!r} is not available in this environment")
            return harness_cls()

        for candidate in auto_select_preference:
            harness_cls = harness_class_by_name[candidate]
            if harness_cls.is_available():
                return harness_cls()

        raise RuntimeError(
            "No benchmark harness available. Checked: "
            + ", ".join(auto_select_preference))

    def __init__(self, settings: _Settings = None):
        if settings is None:
            settings = self.default_settings()
        self.settings = settings

    @abstractmethod
    def measure_runtime_milliseconds(self, fn: BenchMarkedFunction) -> float:
        """Benchmark fn and return execution time in milliseconds."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return whether this harness type can be instantiated in the current environment."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def default_settings(cls) -> _Settings:
        raise NotImplementedError

    def get_settings_dict(self):
        from dataclasses import asdict
        return asdict(self.settings)

    @classmethod
    @abstractmethod
    def _get_harness_name(cls) -> str:
        """Return a short name for this harness type, to be used in command-line args or config objects"""
        raise NotImplementedError

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if ABC not in cls.__bases__:
            cls.harness_name = cls._get_harness_name()
