"""Tests that JAX storage is preserved through all AMSA operations.

Phase 1 JAX integration: every operation that receives JAX-backed
multivector inputs should return JAX-backed outputs — no silent drops
to dense.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from amsa import Algebra
from amsa.ops import (
    add,
    anticommutator_product,
    bulk,
    bulk_dual,
    bulk_norm,
    bulk_norm_squared,
    bulk_normalize,
    commutator_product,
    conjugate,
    dual,
    exp,
    geometric_product,
    inner_product,
    inverse,
    involute,
    left_contraction,
    motor_exp,
    motor_log,
    neg,
    norm,
    norm_squared,
    normalize,
    outer_product,
    poincare_dual,
    poincare_undual,
    regressive_product,
    reverse,
    right_contraction,
    sandwich,
    scalar_product,
    sub,
    undual,
    weight,
    weight_dual,
    weight_norm,
    weight_norm_squared,
)

from ._utils import assert_allclose, assert_mv_allclose

pytest.importorskip("jax")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vga3d():
    return Algebra.vga3d()


@pytest.fixture()
def pga2d():
    return Algebra.pga2d()


@pytest.fixture()
def pga3d():
    return Algebra.pga3d()


# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------


class TestUnaryJAXPreservation:
    def test_reverse_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = reverse(mv)
        assert result.storage_kind == "jax"

    def test_involute_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = involute(mv)
        assert result.storage_kind == "jax"

    def test_conjugate_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = conjugate(mv)
        assert result.storage_kind == "jax"

    def test_neg_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = neg(mv)
        assert result.storage_kind == "jax"

    def test_dual_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = dual(mv)
        assert result.storage_kind == "jax"

    def test_undual_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = undual(mv)
        assert result.storage_kind == "jax"

    def test_poincare_dual_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = poincare_dual(mv)
        assert result.storage_kind == "jax"

    def test_poincare_undual_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = poincare_undual(mv)
        assert result.storage_kind == "jax"

    def test_grade_projection_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = mv.grade(1)
        assert result.storage_kind == "jax"

    def test_scalar_mul_preserves_jax(self, vga3d):
        mv = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        result = 3.0 * mv
        assert result.storage_kind == "jax"


# ---------------------------------------------------------------------------
# Add / Sub
# ---------------------------------------------------------------------------


class TestAddSubJAXPreservation:
    def test_add_jax_jax_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e2": 2.0}, backend="jax")
        b = vga3d.multivector({"e1": 3.0, "e2": -1.0}, backend="jax")
        result = add(a, b)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e1"), 4.0, tol=1e-5)
        assert_allclose(result.component("e2"), 1.0, tol=1e-5)

    def test_sub_jax_jax_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e2": 2.0}, backend="jax")
        b = vga3d.multivector({"e1": 3.0, "e2": -1.0}, backend="jax")
        result = sub(a, b)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e1"), -2.0, tol=1e-5)
        assert_allclose(result.component("e2"), 3.0, tol=1e-5)

    def test_add_jax_scalar_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        result = add(a, 5.0)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e"), 5.0, tol=1e-5)
        assert_allclose(result.component("e1"), 1.0, tol=1e-5)

    def test_sub_jax_scalar_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e": 3.0}, backend="jax")
        result = sub(a, 1.0)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e"), 2.0, tol=1e-5)

    def test_operator_add_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        b = vga3d.multivector({"e1": 2.0}, backend="jax")
        result = a + b
        assert result.storage_kind == "jax"

    def test_operator_sub_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        b = vga3d.multivector({"e1": 2.0}, backend="jax")
        result = a - b
        assert result.storage_kind == "jax"

    def test_mixed_jax_dense_add_falls_back_to_dense(self, vga3d):
        jax_mv = vga3d.multivector({"e1": 1.0}, backend="jax")
        dense_mv = vga3d.multivector({"e1": 2.0}, backend="dense")
        result = add(jax_mv, dense_mv)
        assert result.storage_kind == "dense"
        assert_allclose(result.component("e1"), 3.0, tol=1e-5)


# ---------------------------------------------------------------------------
# Binary products
# ---------------------------------------------------------------------------


class TestBinaryProductJAXPreservation:
    def test_geometric_product_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        b = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="jax")
        result = geometric_product(a, b)
        assert result.storage_kind == "jax"

    def test_outer_product_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        b = vga3d.multivector({"e2": 3.0}, backend="jax")
        result = outer_product(a, b)
        assert result.storage_kind == "jax"

    def test_inner_product_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        b = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="jax")
        result = inner_product(a, b)
        assert result.storage_kind == "jax"

    def test_scalar_product_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        b = vga3d.multivector({"e1": 2.0}, backend="jax")
        result = scalar_product(a, b)
        assert result.storage_kind == "jax"

    def test_left_contraction_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0}, backend="jax")
        b = vga3d.multivector({"e12": 2.0}, backend="jax")
        result = left_contraction(a, b)
        assert result.storage_kind == "jax"

    def test_right_contraction_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e12": 2.0}, backend="jax")
        b = vga3d.multivector({"e1": 1.0}, backend="jax")
        result = right_contraction(a, b)
        assert result.storage_kind == "jax"

    def test_regressive_product_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e12": 1.0, "e23": 2.0}, backend="jax")
        b = vga3d.multivector({"e13": 3.0, "e23": -1.0}, backend="jax")
        result = regressive_product(a, b)
        assert result.storage_kind == "jax"

    def test_commutator_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        b = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="jax")
        result = commutator_product(a, b)
        assert result.storage_kind == "jax"

    def test_anticommutator_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="jax")
        b = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="jax")
        result = anticommutator_product(a, b)
        assert result.storage_kind == "jax"


# ---------------------------------------------------------------------------
# Norms, inverse, sandwich
# ---------------------------------------------------------------------------


class TestNormInverseJAXPreservation:
    def test_norm_squared_preserves_jax(self, vga3d):
        v = vga3d.vector([1.0, 2.0, 3.0], backend="jax")
        result = norm_squared(v)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e"), 14.0, tol=1e-5)

    def test_norm_preserves_jax(self, vga3d):
        v = vga3d.vector([1.0, 2.0, 3.0], backend="jax")
        result = norm(v)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e"), np.sqrt(14.0), tol=1e-5)

    def test_normalize_preserves_jax(self, vga3d):
        v = vga3d.vector([3.0, 4.0, 0.0], backend="jax")
        result = normalize(v)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e1"), 0.6, tol=1e-5)
        assert_allclose(result.component("e2"), 0.8, tol=1e-5)

    def test_inverse_preserves_jax(self, vga3d):
        v = vga3d.vector([1.0, 2.0, 3.0], backend="jax")
        result = inverse(v)
        assert result.storage_kind == "jax"

    def test_sandwich_preserves_jax(self, vga3d):
        v = vga3d.vector([1.0, 0.0, 0.0], backend="jax")
        target = vga3d.vector([0.0, 1.0, 0.0], backend="jax")
        result = sandwich(v, target)
        assert result.storage_kind == "jax"


# ---------------------------------------------------------------------------
# PGA: bulk, weight, normalization
# ---------------------------------------------------------------------------


class TestPGAJAXPreservation:
    def test_bulk_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e1": 1.0, "e2": 2.0, "e0": 4.0}, backend="jax")
        assert bulk(v).storage_kind == "jax"

    def test_weight_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e1": 1.0, "e2": 2.0, "e0": 4.0}, backend="jax")
        assert weight(v).storage_kind == "jax"

    def test_bulk_dual_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e1": 1.0, "e2": 2.0}, backend="jax")
        assert bulk_dual(v).storage_kind == "jax"

    def test_weight_dual_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e0": 4.0}, backend="jax")
        assert weight_dual(v).storage_kind == "jax"

    def test_bulk_norm_squared_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e1": 3.0, "e2": 4.0}, backend="jax")
        result = bulk_norm_squared(v)
        assert result.storage_kind == "jax"

    def test_bulk_norm_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e1": 3.0, "e2": 4.0}, backend="jax")
        result = bulk_norm(v)
        assert result.storage_kind == "jax"

    def test_weight_norm_squared_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e0": 4.0}, backend="jax")
        result = weight_norm_squared(v)
        assert result.storage_kind == "jax"

    def test_weight_norm_preserves_jax(self, pga3d):
        v = pga3d.multivector({"e0": 4.0}, backend="jax")
        result = weight_norm(v)
        assert result.storage_kind == "jax"

    def test_bulk_normalize_preserves_jax(self, pga3d):
        motor = pga3d.multivector({"e": 2.0, "e12": 1.0}, backend="jax")
        result = bulk_normalize(motor)
        assert result.storage_kind == "jax"


# ---------------------------------------------------------------------------
# Exp / Log
# ---------------------------------------------------------------------------


class TestExpLogJAXPreservation:
    def test_exp_simple_preserves_jax(self, vga3d):
        bv = vga3d.multivector({"e12": 0.5}, backend="jax")
        result = exp(bv)
        assert result.storage_kind == "jax"
        # Verify correctness against dense
        bv_dense = vga3d.multivector({"e12": 0.5}, backend="dense")
        assert_mv_allclose(result, bv_dense.exp(), tol=1e-5)

    def test_motor_exp_pga3d_preserves_jax(self, pga3d):
        gen = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05}, backend="jax")
        result = motor_exp(gen)
        assert result.storage_kind == "jax"
        # Verify correctness against dense
        gen_dense = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05}, backend="dense")
        assert_mv_allclose(result, motor_exp(gen_dense), tol=1e-5)

    def test_motor_log_pga2d_preserves_jax(self, pga2d):
        gen = pga2d.multivector({"e12": -0.35, "e01": 0.1, "e02": -0.2}, backend="jax")
        motor = exp(gen)
        result = motor_log(motor)
        assert result.storage_kind == "jax"
        # Round-trip: log(exp(gen)) ≈ gen
        assert_allclose(result.component("e12"), -0.35, tol=1e-5)

    def test_motor_log_pga3d_preserves_jax(self, pga3d):
        gen = pga3d.multivector({"e12": -0.3, "e03": 0.2, "e01": 0.05}, backend="jax")
        motor = motor_exp(gen)
        result = motor_log(motor)
        assert result.storage_kind == "jax"


class TestJAXEagerValidation:
    @pytest.mark.parametrize(
        "invalid_multivector",
        [
            {"e1": 1.0, "e12": 1.0},
            {"e": 1.0, "e1": 1.0},
        ],
    )
    def test_inverse_raises_on_non_scalar_jax_eager(self, vga3d, invalid_multivector):
        mv = vga3d.multivector(invalid_multivector, backend="jax")
        with pytest.raises(ValueError, match="must be scalar-valued"):
            inverse(mv)

    @pytest.mark.parametrize(
        "invalid_multivector",
        [
            {"e1": 1.0, "e23": 1.0},
            {"e": 1.0, "e1": 1.0},
        ],
    )
    def test_exp_raises_on_non_scalar_jax_eager(self, vga3d, invalid_multivector):
        mv = vga3d.multivector(invalid_multivector, backend="jax")
        with pytest.raises(ValueError, match="must be scalar-valued"):
            exp(mv)

    def test_inverse_jit_tracing_preserves_jax(self, vga3d):
        mv = vga3d.vector([1.0, 2.0, 3.0], backend="jax")
        jit_inverse = jax.jit(inverse)
        result = jit_inverse(mv)
        assert result.storage_kind == "jax"
        assert_mv_allclose(result, inverse(mv), tol=1e-5)

    def test_exp_jit_tracing_preserves_jax(self, vga3d):
        bv = vga3d.multivector({"e12": 0.5}, backend="jax")
        jit_exp = jax.jit(exp)
        result = jit_exp(bv)
        assert result.storage_kind == "jax"
        assert_mv_allclose(result, exp(bv), tol=1e-5)

    def test_jax_eager_validation_stress(self, vga3d):
        invalid_inverses = [
            {"e1": 1.0, "e12": 1.0},
            {"e": 1.0, "e1": 1.0},
        ]
        invalid_exps = [
            {"e1": 1.0, "e23": 1.0},
            {"e": 1.0, "e1": 1.0},
        ]
        for invalid in invalid_inverses:
            mv = vga3d.multivector(invalid, backend="jax")
            with pytest.raises(ValueError, match="must be scalar-valued"):
                inverse(mv)
        for invalid in invalid_exps:
            mv = vga3d.multivector(invalid, backend="jax")
            with pytest.raises(ValueError, match="must be scalar-valued"):
                exp(mv)


# ---------------------------------------------------------------------------
# Batched JAX
# ---------------------------------------------------------------------------


class TestBatchedJAXPreservation:
    def test_batched_gp_preserves_jax(self, vga3d):
        a = vga3d.multivector(
            {"e1": np.linspace(0.5, 2.0, 16), "e23": 1.0}, backend="jax"
        )
        b = vga3d.multivector(
            {"e2": 2.0, "e12": np.linspace(-1.0, 1.0, 16)}, backend="jax"
        )
        result = geometric_product(a, b)
        assert result.storage_kind == "jax"
        assert result.batch_shape == (16,)

    def test_batched_add_preserves_jax(self, vga3d):
        a = vga3d.multivector({"e1": np.ones(8)}, backend="jax")
        b = vga3d.multivector({"e1": np.ones(8) * 2}, backend="jax")
        result = add(a, b)
        assert result.storage_kind == "jax"
        assert_allclose(result.component("e1"), np.ones(8) * 3, tol=1e-5)

    def test_batched_norm_preserves_jax(self, vga3d):
        v = vga3d.multivector(
            {"e1": np.ones(4), "e2": np.ones(4) * 2, "e3": np.ones(4) * 3}, backend="jax"
        )
        result = norm(v)
        assert result.storage_kind == "jax"
        assert result.batch_shape == (4,)


# ---------------------------------------------------------------------------
# Correctness parity: JAX results must match dense results
# ---------------------------------------------------------------------------


class TestJAXDenseCorrectness:
    def test_add_values_match(self, vga3d):
        a_d = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="dense")
        b_d = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="dense")
        a_j = a_d.with_storage("jax")
        b_j = b_d.with_storage("jax")
        assert_mv_allclose(add(a_j, b_j), add(a_d, b_d), tol=1e-5)

    def test_sub_values_match(self, vga3d):
        a_d = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="dense")
        b_d = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="dense")
        a_j = a_d.with_storage("jax")
        b_j = b_d.with_storage("jax")
        assert_mv_allclose(sub(a_j, b_j), sub(a_d, b_d), tol=1e-5)

    def test_gp_values_match(self, vga3d):
        a_d = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="dense")
        b_d = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="dense")
        a_j = a_d.with_storage("jax")
        b_j = b_d.with_storage("jax")
        assert_mv_allclose(
            geometric_product(a_j, b_j),
            geometric_product(a_d, b_d),
            tol=1e-5,
        )

    def test_commutator_values_match(self, vga3d):
        a_d = vga3d.multivector({"e1": 1.0, "e23": 2.0}, backend="dense")
        b_d = vga3d.multivector({"e2": 3.0, "e12": -1.0}, backend="dense")
        a_j = a_d.with_storage("jax")
        b_j = b_d.with_storage("jax")
        assert_mv_allclose(
            commutator_product(a_j, b_j),
            commutator_product(a_d, b_d),
            tol=1e-5,
        )

    def test_norm_values_match(self, vga3d):
        v_d = vga3d.vector([1.0, 2.0, 3.0], backend="dense")
        v_j = v_d.with_storage("jax")
        assert_mv_allclose(norm(v_j), norm(v_d), tol=1e-5)
