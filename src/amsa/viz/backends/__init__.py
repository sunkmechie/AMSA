"""Optional visualization backends for :mod:`amsa.viz`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["mpl", "vispy"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
