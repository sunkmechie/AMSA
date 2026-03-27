from __future__ import annotations

from typing import Any

import numpy as np

from amsa.mv import MVArray


def assert_allclose(
    actual: Any,
    desired: Any,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    tol: float | None = None,
    err_msg: str | None = None,
) -> None:
    """
    Assert numeric closeness with a compact tolerance API.

    `tol` is a convenience alias that applies to both `atol` and `rtol`.
    """
    if tol is not None:
        if atol is not None or rtol is not None:
            raise ValueError("Use either `tol` or (`atol`, `rtol`), not both.")
        atol = tol
        rtol = tol

    np.testing.assert_allclose(
        actual,
        desired,
        atol=0.0 if atol is None else atol,
        rtol=0.0 if rtol is None else rtol,
        err_msg="" if err_msg is None else err_msg,
    )


def assert_mv_allclose(
    actual: MVArray,
    desired: MVArray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    tol: float | None = None,
) -> None:
    """Compare multivectors through their dense coefficient arrays."""
    if actual.algebra != desired.algebra:
        raise AssertionError("Multivectors belong to different algebras.")

    actual_dense = actual.as_dense()
    desired_dense = desired.as_dense()
    assert_allclose(actual_dense.values, desired_dense.values, atol=atol, rtol=rtol, tol=tol)
