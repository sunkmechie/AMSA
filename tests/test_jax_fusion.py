"""Tests that AMSA structural models resolve properly as JAX PyTrees.

By registering MVArray and JAXStorage as pytree nodes, users can apply
@jax.jit directly to functions that manipulate multivectors.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from amsa import Algebra
from amsa.ops import inner_product, reverse, sandwich
from tests._utils import assert_mv_allclose


def test_jit_compile_binary_op():
    vga3d = Algebra.vga3d()

    def my_fused_op(a, b):
        return a * b + reverse(a)

    jitted_op = jax.jit(my_fused_op)

    a = vga3d.multivector({"e1": 1.0, "e2": 2.0}, backend="jax")
    b = vga3d.multivector({"e12": -1.0, "e3": 0.5}, backend="jax")

    result_expected = my_fused_op(a, b)
    result_jitted = jitted_op(a, b)

    assert result_jitted.storage_kind == "jax"
    assert_mv_allclose(result_jitted, result_expected, tol=1e-5)


def test_jit_compile_sandwich():
    vga3d = Algebra.vga3d()

    @jax.jit
    def apply_rotor(R, v):
        return sandwich(R, v)

    # 90 degree rotation in e12 plane
    R = vga3d.multivector({"e": np.cos(np.pi / 4), "e12": -np.sin(np.pi / 4)}, backend="jax")
    v = vga3d.multivector({"e1": 1.0}, backend="jax")

    result_expected = sandwich(R, v)
    result_jitted = apply_rotor(R, v)

    assert result_jitted.storage_kind == "jax"
    assert_mv_allclose(result_jitted, result_expected, tol=1e-5)


def test_jit_compile_composite_expression():
    pga2d = Algebra.pga2d()

    @jax.jit
    def complex_metric(a, b, c):
        return inner_product(a * b, c)

    a = pga2d.multivector({"e1": 1.0, "e02": 0.5}, backend="jax")
    b = pga2d.multivector({"e2": -1.0, "e0": 1.5}, backend="jax")
    c = pga2d.multivector({"e12": 0.2, "e01": 0.3}, backend="jax")

    result_expected = inner_product(a * b, c)
    result_jitted = complex_metric(a, b, c)

    assert result_jitted.storage_kind == "jax"
    assert_mv_allclose(result_jitted, result_expected, tol=1e-5)
