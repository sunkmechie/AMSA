# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from math import prod
from typing import Any, Literal

from amsa.mv import MVArray
from amsa.viz.adapters import to_point
from amsa.viz.primitives import VizPrimitive

_logger = logging.getLogger(__name__)

type BackendKind = Literal["auto", "mpl", "vispy"]

# Internal state
_ACTIVE_BACKEND: BackendKind = "auto"

def use_backend(kind: BackendKind) -> None:
    """Manually set the visualization backend."""
    global _ACTIVE_BACKEND
    if kind not in ("auto", "mpl", "vispy"):
        raise ValueError(f"Unsupported backend: {kind}")
    _ACTIVE_BACKEND = kind

def _resolve_backend(mv_hint: MVArray | None = None) -> Literal["mpl", "vispy"]:
    """Select the best available backend based on environment and data size."""
    if _ACTIVE_BACKEND != "auto":
        return _ACTIVE_BACKEND  # type: ignore

    try:
        import vispy
        # Use VisPy for large batches (>100 elements)
        if mv_hint is not None and prod(mv_hint.batch_shape) > 100:
            return "vispy"
        # For a clean first impression, we default to MPL for small batches
        return "mpl"
    except ImportError:
        return "mpl"

def plot(primitive: VizPrimitive, **kwargs: Any) -> Any:
    """
    Plot a geometric primitive using the active backend.
    """
    backend = kwargs.pop("backend", _resolve_backend())
    if backend == "vispy":
        from amsa.viz.backends.vispy import plot as vispy_plot
        raise RuntimeError("vispy backend requires an AMSAScene. Use viz.view() or AMSAScene.add().")
    else:
        import matplotlib.pyplot as plt
        from amsa.viz.backends.mpl import plot as mpl_plot
        ax = kwargs.pop("ax", plt.gca())
        return mpl_plot(ax, primitive, **kwargs)

def show() -> None:
    """
    Start the visualization event loop for the active backend.
    """
    backend = _resolve_backend()
    if backend == "vispy":
        from amsa.viz.backends.vispy import show as vispy_show
        vispy_show()
    else:
        import matplotlib.pyplot as plt
        plt.show()

def view(mv: MVArray, backend: BackendKind = "auto", **kwargs: Any) -> Any:
    """
    High-level entry point to visualize a multivector array.
    """
    resolved_backend = backend if backend != "auto" else _resolve_backend(mv_hint=mv)
    primitive = to_point(mv)

    if resolved_backend == "vispy":
        from amsa.viz.backends.vispy import AMSAScene
        scene = AMSAScene(title=kwargs.get("title", "AMSA View"))
        scene.add(primitive, **kwargs)
        return scene
    else:
        import matplotlib.pyplot as plt
        from amsa.viz.backends.mpl import plot as mpl_plot
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
        title = kwargs.pop("title", "AMSA View")
        if primitive.position.shape[-1] == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        mpl_plot(ax, primitive, **kwargs)
        ax.set_title(title)
        return ax
