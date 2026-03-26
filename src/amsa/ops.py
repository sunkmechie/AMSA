from __future__ import annotations

from amsa.mv import MVArray


def ensure_compatible(lhs: MVArray, rhs: MVArray) -> None:
    """Validate that two multivectors share algebra and layout metadata."""
    if lhs.algebra != rhs.algebra:
        raise ValueError("Multivectors belong to different algebras.")
    if lhs.layout != rhs.layout:
        raise ValueError("Multivectors must share the same layout in the initial scaffold.")
