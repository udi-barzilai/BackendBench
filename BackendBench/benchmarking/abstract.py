from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic, Optional
from dataclasses import dataclass


__all__ = 'BenchmarkHarness', 'BenchMarkedFunction', 'settings_class'


BenchMarkedFunction = Callable[[], Any]

settings_class = dataclass(kw_only=True)  # <- decorator
_Settings = TypeVar('_Settings')


class BenchmarkHarness(Generic[_Settings], ABC):
    @classmethod
    def create(cls, name: str) -> 'BenchmarkHarness':
        """Create a harness by name."""
        from ._registry import harness_class_by_name

        if name not in harness_class_by_name:
            raise ValueError(
                f"Unknown harness: {name!r}. Available: {list(harness_class_by_name)}")
        return harness_class_by_name[name]()

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
