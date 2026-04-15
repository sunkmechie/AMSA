# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
VisPy backend for AMSA high-performance visualization.
"""

from __future__ import annotations

from typing import Any

try:
    import vispy
    from vispy import scene
except ImportError as exc:
    raise ImportError(
        "vispy is required for amsa.viz.backends.vispy. "
        "Install it via `pip install 'amsa[viz]'`."
    ) from exc

from amsa.viz.primitives import VizPrimitive

def plot(canvas: Any, primitive: VizPrimitive, **kwargs: Any) -> None:
    """
    Plots a viz primitive onto a VisPy canvas/scene.
    
    To be implemented in Phase 2.
    """
    raise NotImplementedError("VisPy primitive plotting is not yet implemented (Phase 2).")

def show() -> None:
    """
    Starts the VisPy event loop.
    """
    vispy.app.run()
