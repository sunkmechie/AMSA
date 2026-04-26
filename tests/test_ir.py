from __future__ import annotations

import numpy as np
import pytest

from amsa import Algebra
from amsa.backends.numpy import NumpyBackend
from amsa.ir import (
    IRStep,
    ProductIR,
    SequenceIR,
    build_product_ir,
    build_unary_ir,
    clear_backends,
    get_backend,
    has_backend,
    list_backends,
    output_layout_from_product_ir,
    output_layout_from_unary_ir,
    register_backend,
    set_default_backend,
    unregister_backend,
)
from amsa.plans import plan_binary_product

from ._utils import assert_allclose


@pytest.fixture(autouse=True)
def _clean_backends():
    """Ensure backend registry starts and ends clean."""
    from amsa.ir import _BACKENDS, _DEFAULT_BACKEND
    saved_backends = dict(_BACKENDS)
    saved_default = _DEFAULT_BACKEND

    clear_backends()
    yield
    clear_backends()
    for name, executor in saved_backends.items():
        register_backend(name, executor)
    if saved_default:
        set_default_backend(saved_default)



class TestBuildProductIR:
    def test_geometric_product_vectors_vga2d(self):
        alg = Algebra.vga2d()
        u = alg.vector([1.0, 2.0])
        v = alg.vector([3.0, -4.0])
        plan = plan_binary_product(u.layout, v.layout, "geometric")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)

        assert isinstance(ir, ProductIR)
        assert ir.kind == "geometric"
        assert ir.lhs_width == 2
        assert ir.rhs_width == 2
        assert len(ir.terms) == len(plan.terms)
        assert ir.out_blades == plan.output_blades

    def test_outer_product_vectors_vga3d(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])
        plan = plan_binary_product(u.layout, v.layout, "outer")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)

        assert ir.kind == "outer"
        assert ir.out_blades == plan.output_blades
        for pt, it in zip(plan.terms, ir.terms, strict=False):
            assert it.lhs_col == pt.lhs_index
            assert it.rhs_col == pt.rhs_index
            assert it.coefficient == pt.coefficient

    def test_empty_product_ir(self):
        alg = Algebra.vga3d()
        b = alg.bivector([1.0, 0.0, 0.0])
        t = alg.trivector([1.0])
        plan = plan_binary_product(b.layout, t.layout, "outer")
        ir = build_product_ir(plan, b.storage_kind, t.storage_kind)

        assert len(ir.terms) == 0
        assert ir.out_blades == ()
        assert ir.out_width == 0

    def test_all_operator_kinds(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])
        for kind in ("geometric", "outer", "inner", "scalar"):
            plan = plan_binary_product(u.layout, v.layout, kind)
            ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
            assert ir.kind == kind
            assert ir.out_blades == plan.output_blades

    def test_storage_kind_propagated(self):
        alg = Algebra.vga2d()
        u = alg.vector([1.0, 2.0])
        v = alg.vector([3.0, 4.0])
        plan = plan_binary_product(u.layout, v.layout, "geometric")

        ir_dense = build_product_ir(plan, "dense", "dense")
        assert ir_dense.lhs_storage == "dense"
        assert ir_dense.rhs_storage == "dense"

        ir_csr = build_product_ir(plan, "csr", "csr")
        assert ir_csr.lhs_storage == "csr"
        assert ir_csr.rhs_storage == "csr"




class TestBuildUnaryIR:
    @pytest.fixture(autouse=True)
    def _register_numpy(self):
        register_backend("numpy", NumpyBackend())

    def test_reverse_signs(self):
        alg = Algebra.vga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "reverse")

        assert ir.kind == "reverse"
        assert ir.out_blades == blades
        assert ir.input_width == len(blades)
        assert ir.permutation is None
        assert ir.is_permutation is False

        # Verify against actual reverse
        mv = alg.multivector({b: float(i) for i, b in enumerate(blades)})
        actual_rev = mv.reverse()
        for i, blade in enumerate(blades):
            expected = actual_rev.component(blade)
            actual = ir.weights[i] * mv.component(blade)
            assert np.isclose(actual, expected)

    def test_involute_signs(self):
        alg = Algebra.vga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "involute")

        assert ir.kind == "involute"
        assert ir.permutation is None

        mv = alg.multivector({b: float(i) for i, b in enumerate(blades)})
        actual_inv = mv.involute()
        for i, blade in enumerate(blades):
            expected = actual_inv.component(blade)
            actual = ir.weights[i] * mv.component(blade)
            assert np.isclose(actual, expected)

    def test_conjugate_signs(self):
        alg = Algebra.vga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "conjugate")

        assert ir.kind == "conjugate"
        assert ir.permutation is None

        mv = alg.multivector({b: float(i) for i, b in enumerate(blades)})
        actual_conj = mv.conjugate()
        for i, blade in enumerate(blades):
            expected = actual_conj.component(blade)
            actual = ir.weights[i] * mv.component(blade)
            assert np.isclose(actual, expected)

    def test_dual_permutation_vga3d(self):
        alg = Algebra.vga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "dual")

        assert ir.kind == "dual"
        assert ir.is_permutation
        assert ir.permutation is not None
        assert len(ir.permutation) == len(blades)

        # Cross-validate against actual dual
        mv = alg.multivector({b: float(i) for i, b in enumerate(blades)})
        actual_dual = mv.dual()
        for out_col, target_blade in enumerate(ir.out_blades):
            src_col = ir.permutation[out_col]
            src_blade = blades[src_col]
            expected = actual_dual.component(target_blade)
            actual = ir.weights[out_col] * mv.component(src_blade)
            assert np.isclose(actual, expected)

    def test_poincare_dual_pga3d(self):
        alg = Algebra.pga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "poincare_dual")

        assert ir.kind == "poincare_dual"
        assert ir.is_permutation
        assert ir.permutation is not None

        # Create a dense MV to validate the IR's source column indexing
        mv = alg.multivector({b: float(b) for b in blades})
        actual_pd = mv.poincare_dual()
        for out_col, target_blade in enumerate(ir.out_blades):
            src_col = ir.permutation[out_col]
            src_blade = mv.layout.blades[src_col]
            expected = actual_pd.component(target_blade)
            actual = ir.weights[out_col] * mv.component(src_blade)
            assert np.isclose(actual, expected)

    def test_poincare_undual_pga3d(self):
        alg = Algebra.pga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "poincare_undual")

        assert ir.kind == "poincare_undual"
        assert ir.is_permutation

        # Create a dense MV to validate the IR's source column indexing
        mv = alg.multivector({b: float(b) for b in blades})
        actual_pu = mv.poincare_undual()
        for out_col, target_blade in enumerate(ir.out_blades):
            src_col = ir.permutation[out_col]
            src_blade = mv.layout.blades[src_col]
            expected = actual_pu.component(target_blade)
            actual = ir.weights[out_col] * mv.component(src_blade)
            assert np.isclose(actual, expected)

    def test_metric_dual_on_degenerate_raises(self):
        alg = Algebra.pga3d()
        blades = tuple(range(alg.spec.blade_count))
        with pytest.raises(ValueError, match="invertible pseudoscalar"):
            build_unary_ir(blades, alg.spec, "dual")

    def test_empty_blades(self):
        alg = Algebra.vga2d()
        ir = build_unary_ir((), alg.spec, "reverse")

        assert ir.input_width == 0
        assert ir.out_width == 0
        assert ir.out_blades == ()
        assert ir.weights == ()



class TestLayoutReconstruction:
    def test_product_ir_layout_matches_plan(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])
        plan = plan_binary_product(u.layout, v.layout, "geometric")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)

        plan_layout = plan.output_layout()
        ir_layout = output_layout_from_product_ir(ir, alg.spec)

        assert plan_layout.blades == ir_layout.blades
        assert plan_layout.kind == ir_layout.kind
        assert plan_layout.name == ir_layout.name

    def test_unary_ir_dual_layout(self):
        alg = Algebra.vga3d()
        blades = tuple(range(alg.spec.blade_count))
        ir = build_unary_ir(blades, alg.spec, "dual")
        layout = output_layout_from_unary_ir(ir, alg.spec)

        expected = tuple(sorted(blade ^ alg.spec.pseudoscalar_blade for blade in blades))
        assert layout.blades == expected
        assert layout.kind == "dense"

    def test_unary_ir_sparse_layout(self):
        alg = Algebra.vga3d()
        blades = (0, 3, 7)
        ir = build_unary_ir(blades, alg.spec, "reverse")
        layout = output_layout_from_unary_ir(ir, alg.spec)

        assert layout.blades == blades
        assert layout.kind == "sparse"



class TestBackendRegistry:
    def test_initially_empty(self):
        assert list_backends() == ()
        assert not has_backend("numpy")

    def test_register_first_becomes_default(self):
        class Stub:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        register_backend("stub", Stub())
        assert list_backends() == ("stub",)
        assert get_backend() is not None

    def test_second_register_preserves_default(self):
        class Stub1:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        class Stub2:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        s1, s2 = Stub1(), Stub2()
        register_backend("s1", s1)
        register_backend("s2", s2)
        assert get_backend() is s1
        assert list_backends() == ("s1", "s2")

    def test_set_default_backend(self):
        class Stub1:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        class Stub2:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        s1, s2 = Stub1(), Stub2()
        register_backend("s1", s1)
        register_backend("s2", s2)
        set_default_backend("s2")
        assert get_backend() is s2

    def test_get_backend_unknown_raises(self):
        with pytest.raises(KeyError, match="is not registered"):
            get_backend("nonexistent")

    def test_set_default_backend_unknown_raises(self):
        with pytest.raises(KeyError, match="is not registered"):
            set_default_backend("nonexistent")

    def test_get_backend_raises_when_no_default(self):
        with pytest.raises(RuntimeError, match="No default backend"):
            get_backend()

    def test_unregister_backend(self):
        class Stub:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        register_backend("stub", Stub())
        assert has_backend("stub")
        unregister_backend("stub")
        assert not has_backend("stub")
        assert list_backends() == ()

    def test_unregister_default_clears_default(self):
        class Stub:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        register_backend("stub", Stub())
        unregister_backend("stub")
        with pytest.raises(RuntimeError, match="No default backend"):
            get_backend()

    def test_unregister_non_default_preserves_default(self):
        class Stub1:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        class Stub2:
            def execute_product(self, lhs, rhs, ir_): pass
            def execute_unary(self, m, i): pass
            def execute_sequence(self, inp, i): pass

        s1, s2 = Stub1(), Stub2()
        register_backend("s1", s1)
        register_backend("s2", s2)
        unregister_backend("s2")
        assert get_backend() is s1



class TestNumpyBackendExecution:
    @pytest.fixture(autouse=True)
    def _register_numpy(self):
        register_backend("numpy", NumpyBackend())

    def test_geometric_product(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])

        backend = get_backend("numpy")
        plan = plan_binary_product(u.layout, v.layout, "geometric")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
        result = backend.execute_product(u, v, ir)

        # Compare with direct ops result
        expected = u * v
        np.testing.assert_allclose(result.values, expected.values)

    def test_outer_product(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])

        backend = get_backend("numpy")
        plan = plan_binary_product(u.layout, v.layout, "outer")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
        result = backend.execute_product(u, v, ir)

        expected = u ^ v
        np.testing.assert_allclose(result.values, expected.values)

    def test_inner_product(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])

        backend = get_backend("numpy")
        plan = plan_binary_product(u.layout, v.layout, "inner")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
        result = backend.execute_product(u, v, ir)

        expected = u | v
        np.testing.assert_allclose(result.values, expected.values)

    def test_reverse(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "reverse")
        result = backend.execute_unary(mv, ir)

        expected = mv.reverse()
        np.testing.assert_allclose(result.values, expected.values)

    def test_involute(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "involute")
        result = backend.execute_unary(mv, ir)

        expected = mv.involute()
        np.testing.assert_allclose(result.values, expected.values)

    def test_conjugate(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "conjugate")
        result = backend.execute_unary(mv, ir)

        expected = mv.conjugate()
        np.testing.assert_allclose(result.values, expected.values)

    def test_dual_vga3d(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "dual")
        result = backend.execute_unary(mv, ir)

        expected = mv.dual()
        np.testing.assert_allclose(result.values, expected.values)

    def test_undual_vga3d(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "undual")
        result = backend.execute_unary(mv, ir)

        expected = mv.undual()
        np.testing.assert_allclose(result.values, expected.values)

    def test_poincare_dual_pga3d(self):
        alg = Algebra.pga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 3: 4.0, 7: 8.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "poincare_dual")
        result = backend.execute_unary(mv, ir)

        expected = mv.poincare_dual()
        np.testing.assert_allclose(result.values, expected.values)

    def test_poincare_undual_pga3d(self):
        alg = Algebra.pga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 3: 4.0, 7: 8.0})

        backend = get_backend("numpy")
        ir = build_unary_ir(mv.layout.blades, alg.spec, "poincare_undual")
        result = backend.execute_unary(mv, ir)

        expected = mv.poincare_undual()
        np.testing.assert_allclose(result.values, expected.values)

    def test_sparse_layouts(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])

        backend = get_backend("numpy")
        plan = plan_binary_product(u.layout, v.layout, "geometric")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
        result = backend.execute_product(u, v, ir)

        expected = u * v
        np.testing.assert_allclose(result.values, expected.values)

    def test_csr_storage(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0]).with_storage("csr")
        v = alg.vector([4.0, -5.0, 6.0]).with_storage("csr")

        backend = get_backend("numpy")
        plan = plan_binary_product(u.layout, v.layout, "geometric")
        ir = build_product_ir(plan, u.storage_kind, v.storage_kind)
        result = backend.execute_product(u, v, ir)

        expected = u * v
        np.testing.assert_allclose(result.values, expected.values)

    def test_empty_product(self):
        alg = Algebra.vga3d()
        b = alg.bivector([1.0, 0.0, 0.0])
        t = alg.trivector([1.0])

        backend = get_backend("numpy")
        plan = plan_binary_product(b.layout, t.layout, "outer")
        ir = build_product_ir(plan, b.storage_kind, t.storage_kind)
        result = backend.execute_product(b, t, ir)

        assert result.layout.size == 0

    def test_sequence_execution_supports_add_scale_and_row_scale(self):
        alg = Algebra.vga2d()
        lhs = alg.vector([[1.0, 2.0], [3.0, 4.0]])
        rhs = alg.vector([[5.0, 6.0], [7.0, 8.0]])
        backend = get_backend("numpy")

        add_ir = SequenceIR(
            name="add",
            inputs=("lhs", "rhs"),
            steps=(
                IRStep(
                    kind="add",
                    operands=("lhs", "rhs"),
                    ir=None,
                    output="sum",
                ),
            ),
            result="sum",
        )
        added = backend.execute_sequence({"lhs": lhs, "rhs": rhs}, add_ir)
        np.testing.assert_allclose(added.values, (lhs + rhs).values)

        scale_ir = SequenceIR(
            name="scale",
            inputs=("input",),
            steps=(
                IRStep(
                    kind="scale",
                    operands=("input",),
                    ir=None,
                    output="scaled",
                    metadata={"factor": 2.0},
                ),
            ),
            result="scaled",
        )
        scaled = backend.execute_sequence({"input": lhs}, scale_ir)
        np.testing.assert_allclose(scaled.values, (2.0 * lhs).values)

        row_scale_ir = SequenceIR(
            name="row_scale",
            inputs=("input",),
            steps=(
                IRStep(
                    kind="row_scale",
                    operands=("input",),
                    ir=None,
                    output="scaled",
                    metadata={"scales": np.array([2.0, 3.0])},
                ),
            ),
            result="scaled",
        )
        row_scaled = backend.execute_sequence({"input": lhs}, row_scale_ir)
        np.testing.assert_allclose(row_scaled.values, np.array([[2.0, 4.0], [9.0, 12.0]]))

    def test_sequence_execution_supports_scalar_primitives(self):
        alg = Algebra.vga2d()
        mv = alg.multivector({"e": np.array([4.0, 9.0]), "e1": np.array([3.0, 4.0])})
        backend = get_backend("numpy")

        ir = SequenceIR(
            name="scalar_primitives",
            inputs=("input",),
            steps=(
                IRStep(
                    kind="component",
                    operands=("input",),
                    ir=None,
                    output="scalar_values",
                    metadata={"blade": 0},
                ),
                IRStep(
                    kind="elementwise",
                    operands=("scalar_values",),
                    ir=None,
                    output="roots",
                    metadata={"function": "sqrt"},
                ),
                IRStep(
                    kind="scalar_mv_from_array",
                    operands=("input", "roots"),
                    ir=None,
                    output="result",
                ),
            ),
            result="result",
        )

        result = backend.execute_sequence({"input": mv}, ir)

        assert result.layout.blades == (0,)
        assert_allclose(result.values, np.array([[2.0], [3.0]]))

    def test_sequence_execution_supports_coefficient_reduction_and_blade_construction(self):
        alg = Algebra.vga2d()
        mv = alg.vector([[3.0, 4.0], [5.0, 12.0]])
        backend = get_backend("numpy")

        ir = SequenceIR(
            name="coefficient_reduction",
            inputs=("input",),
            steps=(
                IRStep(
                    kind="coefficient_norm_squared",
                    operands=("input",),
                    ir=None,
                    output="squares",
                ),
                IRStep(
                    kind="elementwise",
                    operands=("squares",),
                    ir=None,
                    output="magnitudes",
                    metadata={"function": "sqrt"},
                ),
                IRStep(
                    kind="single_blade_mv_from_array",
                    operands=("input", "magnitudes"),
                    ir=None,
                    output="result",
                    metadata={"blade": 1},
                ),
            ),
            result="result",
        )

        result = backend.execute_sequence({"input": mv}, ir)

        assert result.layout.blades == (1,)
        assert_allclose(result.values, np.array([[5.0], [13.0]]))

    def test_sequence_execution_supports_exp_coefficient_kernel(self):
        backend = get_backend("numpy")
        scalars = np.array([1.0, -1.0, 0.0])
        ir = SequenceIR(
            name="exp_coefficients",
            inputs=("scalars",),
            steps=(
                IRStep(
                    kind="exp_coefficients",
                    operands=("scalars",),
                    ir=None,
                    output="coefficients",
                ),
            ),
            result="coefficients",
        )

        scalar_coeff, linear_coeff = backend.execute_sequence({"scalars": scalars}, ir)

        assert_allclose(
            scalar_coeff,
            np.array([np.cosh(1.0), np.cos(1.0), 1.0]),
        )
        assert_allclose(
            linear_coeff,
            np.array([np.sinh(1.0), np.sin(1.0), 1.0]),
        )

    def test_sequence_execution_supports_motor_exp_coefficient_kernel(self):
        backend = get_backend("numpy")
        ir = SequenceIR(
            name="motor_exp_coefficients",
            inputs=("scalar", "pseudoscalar"),
            steps=(
                IRStep(
                    kind="motor_exp_coefficients",
                    operands=("scalar", "pseudoscalar"),
                    ir=None,
                    output="coefficients",
                ),
            ),
            result="coefficients",
        )

        scalar, pseudo, linear, dual_linear = backend.execute_sequence(
            {
                "scalar": np.array([0.0, -0.09]),
                "pseudoscalar": np.array([0.6, -0.12]),
            },
            ir,
        )

        assert_allclose(scalar[0], 1.0)
        assert_allclose(pseudo[0], 0.3)
        assert_allclose(linear[0], 1.0)
        assert_allclose(dual_linear[0], 0.1, tol=1e-15)
        assert_allclose(scalar[1], np.cos(0.3), tol=1e-15)
        assert_allclose(linear[1], np.sin(0.3) / 0.3, tol=1e-15)

    def test_sequence_execution_supports_predicate_kernel(self):
        backend = get_backend("numpy")
        ir = SequenceIR(
            name="predicate",
            inputs=("lhs", "rhs"),
            steps=(
                IRStep(
                    kind="predicate",
                    operands=("lhs", "rhs"),
                    ir=None,
                    output="result",
                    metadata={"function": "allclose"},
                ),
            ),
            result="result",
        )

        assert backend.execute_sequence(
            {"lhs": np.array([1.0, 2.0]), "rhs": np.array([1.0, 2.0])},
            ir,
        )

    def test_sequence_execution_supports_motor_log_coefficient_kernels(self):
        backend = get_backend("numpy")

        simple_ir = SequenceIR(
            name="simple_bivector_log_coefficients",
            inputs=("scalar", "square"),
            steps=(
                IRStep(
                    kind="simple_bivector_log_coefficients",
                    operands=("scalar", "square"),
                    ir=None,
                    output="coefficients",
                ),
            ),
            result="coefficients",
        )
        simple = backend.execute_sequence(
            {"scalar": np.array([1.0, 2.0]), "square": np.array([0.0, -1.0])},
            simple_ir,
        )
        assert_allclose(simple, np.array([1.0, np.arctan2(1.0, 2.0)]))

        pga3d_ir = SequenceIR(
            name="pga3d_motor_log_coefficients",
            inputs=("scalar", "pseudoscalar", "sine"),
            steps=(
                IRStep(
                    kind="pga3d_motor_log_coefficients",
                    operands=("scalar", "pseudoscalar", "sine"),
                    ir=None,
                    output="coefficients",
                ),
            ),
            result="coefficients",
        )
        alpha, beta = backend.execute_sequence(
            {
                "scalar": np.array([1.0, np.cos(0.3)]),
                "pseudoscalar": np.array([0.0, -0.2 * np.sin(0.3)]),
                "sine": np.array([0.0, np.sin(0.3)]),
            },
            pga3d_ir,
        )

        assert_allclose(alpha[0], 0.0)
        assert_allclose(beta[0], 0.0)
        assert_allclose(alpha[1], 0.3 / np.sin(0.3))
        assert_allclose(
            beta[1],
            0.2 * (1.0 - (0.3 * np.cos(0.3) / np.sin(0.3))) / np.sin(0.3),
        )

class TestEndToEndOps:
    """Verify that public ops route through the IR backend correctly."""

    @pytest.fixture(autouse=True)
    def _register_numpy(self):
        register_backend("numpy", NumpyBackend())

    def test_all_binary_products(self):
        alg = Algebra.vga3d()
        u = alg.vector([1.0, 2.0, 3.0])
        v = alg.vector([4.0, -5.0, 6.0])

        gp = u * v
        op = u ^ v
        ip = u | v
        sp = u.scalar_product(v)
        u.left_contract(v)
        u.right_contract(v)

        # All should produce non-empty results
        assert gp.layout.size > 0
        assert op.layout.size > 0
        assert ip.layout.size > 0
        assert sp.layout.size >= 0

    def test_unary_ops(self):
        alg = Algebra.vga3d()
        mv = alg.multivector({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 7: 5.0})

        rev = mv.reverse()
        inv = mv.involute()
        conj = mv.conjugate()
        dual = mv.dual()
        und = mv.undual()

        # Output layouts should match expected
        assert rev.layout.blades == mv.layout.blades
        assert inv.layout.blades == mv.layout.blades
        assert conj.layout.blades == mv.layout.blades
        assert len(dual.layout.blades) == len(mv.layout.blades)
        assert len(und.layout.blades) == len(mv.layout.blades)

    def test_batched_product(self):
        alg = Algebra.vga2d()
        u = alg.vector([[1.0, 2.0], [3.0, 4.0]])
        v = alg.vector([[5.0, 6.0], [7.0, 8.0]])

        gp = u * v
        assert gp.batch_shape == (2,)

    def test_mixed_storage_product(self):
        alg = Algebra.vga2d()
        u = alg.vector([1.0, 2.0])
        v = alg.vector([3.0, 4.0]).with_storage("csr")

        gp = u * v
        expected = alg.vector([1.0, 2.0]) * alg.vector([3.0, 4.0])
        np.testing.assert_allclose(gp.values, expected.values)
