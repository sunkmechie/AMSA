import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import amsa  # noqa: E402
from amsa.backends.jax import JAXBackend  # noqa: E402
from amsa.backends.numpy import NumpyBackend  # noqa: E402
from amsa.ir import IRStep, SequenceIR, clear_backends, init, register_backend  # noqa: E402
from amsa.storage import CSRStorage  # noqa: E402
from tests._utils import assert_allclose  # noqa: E402


def test_jax_backend_registration():
    """Test JAX backend registration behavior."""
    from amsa.ir import has_backend, register_backend
    if not has_backend("jax"):
        register_backend("jax", JAXBackend())
    assert has_backend("jax")


def test_jax_backend_basic_operations():
    """Test that JAX backend executes basic operations correctly."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    # Test with NumPy backend
    init(use="cpu")
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 2.0])
    v = alg.vector([3.0, -4.0])
    numpy_result = u * v
    
    # Test with JAX backend
    init(use="gpu")
    alg2 = amsa.Algebra.vga2d()
    u2 = alg2.vector([1.0, 2.0])
    v2 = alg2.vector([3.0, -4.0])
    jax_result = u2 * v2
    
    # Convert JAX result to numpy for comparison
    jax_values = np.asarray(jax_result.values)
    numpy_values = np.asarray(numpy_result.values)
    
    assert_allclose(jax_values, numpy_values, rtol=1e-5)


def test_jax_backend_addition():
    """Test addition with JAX backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    init(use="cpu")
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 2.0])
    v = alg.vector([3.0, 4.0])
    numpy_result = u + v
    
    init(use="gpu")
    alg2 = amsa.Algebra.vga2d()
    u2 = alg2.vector([1.0, 2.0])
    v2 = alg2.vector([3.0, 4.0])
    jax_result = u2 + v2
    
    assert_allclose(
        np.asarray(jax_result.values),
        np.asarray(numpy_result.values),
        rtol=1e-5
    )


def test_jax_backend_outer_product():
    """Test outer product with JAX backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    init(use="cpu")
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 0.0])
    v = alg.vector([0.0, 1.0])
    numpy_result = u ^ v
    
    init(use="gpu")
    alg2 = amsa.Algebra.vga2d()
    u2 = alg2.vector([1.0, 0.0])
    v2 = alg2.vector([0.0, 1.0])
    jax_result = u2 ^ v2
    
    assert_allclose(
        np.asarray(jax_result.values),
        np.asarray(numpy_result.values),
        rtol=1e-5
    )


def test_jax_backend_scalar_operations():
    """Test scalar operations with JAX backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    init(use="cpu")
    alg = amsa.Algebra.vga2d()
    mv = alg.vector([1.0, 2.0])
    numpy_result = 2.0 * mv
    
    init(use="gpu")
    alg2 = amsa.Algebra.vga2d()
    mv2 = alg2.vector([1.0, 2.0])
    jax_result = 2.0 * mv2
    
    assert_allclose(
        np.asarray(jax_result.values),
        np.asarray(numpy_result.values),
        rtol=1e-5
    )


def test_device_selection_with_jax():
    """Test that amsa.init(use="gpu") selects JAX backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    init(use="gpu")
    from amsa.ir import get_device
    assert get_device() == "gpu"
    
    # Verify operations work with GPU selection
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 2.0])
    v = alg.vector([3.0, -4.0])
    result = u * v
    assert result is not None


def test_device_selection_cpu_fallback():
    """Test that amsa.init(use="cpu") selects NumPy backend."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    
    init(use="cpu")
    from amsa.ir import get_device
    assert get_device() == "cpu"


def test_mvarray_pytree_flatten_uses_coefficients_as_only_leaf():
    """Dense MVArray pytrees should expose coefficients and keep metadata static."""
    alg = amsa.Algebra.vga2d()
    mv = alg.vector([1.0, 2.0])

    leaves, treedef = jax.tree_util.tree_flatten(mv)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert leaves[0].shape == (2,)
    assert restored.algebra == mv.algebra
    assert restored.layout == mv.layout
    assert_allclose(np.asarray(restored.values), np.asarray(mv.values))


def test_mvarray_pytree_jit_round_trip():
    """JIT should round-trip a dense MVArray without unpacking metadata manually."""
    alg = amsa.Algebra.vga2d()
    mv = alg.vector(jnp.array([1.0, 2.0]))

    round_tripped = jax.jit(lambda value: value)(mv)

    assert round_tripped.algebra == mv.algebra
    assert round_tripped.layout == mv.layout
    assert_allclose(np.asarray(round_tripped.values), np.asarray(mv.values))


def test_mvarray_pytree_vmap_identity_preserves_metadata():
    """VMAP should map coefficient leaves while preserving static algebra/layout metadata."""
    alg = amsa.Algebra.vga2d()
    mv = alg.vector(jnp.array([[1.0, 2.0], [3.0, 4.0]]))

    mapped = jax.vmap(lambda value: value)(mv)

    assert mapped.algebra == mv.algebra
    assert mapped.layout == mv.layout
    assert mapped.batch_shape == mv.batch_shape
    assert_allclose(np.asarray(mapped.values), np.asarray(mv.values))


def test_mvarray_pytree_rejects_csr_storage_for_jax():
    """CSR-on-JAX is intentionally deferred by the traceability contract."""
    alg = amsa.Algebra.vga2d()
    layout = alg.grade_layout(1)
    mv = amsa.MVArray(
        algebra=alg.spec,
        layout=layout,
        storage=CSRStorage(
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            np.array([0, 2]),
            batch_shape=(),
            width=layout.size,
        ),
    )

    with pytest.raises(TypeError, match="dense MVArray storage only"):
        jax.tree_util.tree_flatten(mv)


def test_jax_jit_dense_add_sub_and_norm_squared():
    """Dense add/sub/norm-squared should trace after storage-local cleanup."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga2d()
    lhs = alg.vector(jnp.array([1.0, 2.0]))
    rhs = alg.vector(jnp.array([3.0, -4.0]))

    def add_sub_norm(a, b):
        combined = (a + b) - b
        return amsa.norm_squared(combined).values

    try:
        actual = jax.jit(add_sub_norm)(lhs, rhs)
    finally:
        init(use="cpu")

    assert_allclose(np.asarray(actual), np.array([5.0]), rtol=1e-5)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda lhs, rhs: lhs * rhs, np.array([-5.0, -10.0])),
        (lambda lhs, rhs: lhs ^ rhs, np.array([-10.0])),
        (lambda lhs, rhs: lhs | rhs, np.array([-5.0])),
        (amsa.scalar_product, np.array([-5.0])),
        (amsa.left_contraction, np.array([-5.0])),
        (amsa.right_contraction, np.array([-5.0])),
    ],
)
def test_jax_jit_dense_binary_products(operation, expected):
    """Dense binary products should trace through the public operation layer."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga2d()
    lhs = alg.vector(jnp.array([1.0, 2.0]))
    rhs = alg.vector(jnp.array([3.0, -4.0]))

    try:
        actual = jax.jit(lambda a, b: operation(a, b).values)(lhs, rhs)
    finally:
        init(use="cpu")

    assert_allclose(np.asarray(actual), expected, rtol=1e-5)


def test_jax_jit_dense_regressive_product():
    """Regressive products should trace for dense nondegenerate algebras."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga3d()
    lhs = alg.bivector(jnp.array([1.0, -2.0, 0.5]))
    rhs = alg.bivector(jnp.array([0.25, 3.0, -1.0]))

    try:
        actual = jax.jit(lambda a, b: amsa.regressive_product(a, b).values)(lhs, rhs)
    finally:
        init(use="cpu")

    init(use="cpu")
    expected = amsa.regressive_product(
        alg.bivector(np.array([1.0, -2.0, 0.5])),
        alg.bivector(np.array([0.25, 3.0, -1.0])),
    )
    assert_allclose(np.asarray(actual), np.asarray(expected.values), rtol=1e-5)


def test_jax_jit_dense_layout_product_preserves_full_output_shape():
    """Dense layouts should trace with the full layout width when support is dense."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga2d()
    lhs = alg.multivector(jnp.array([1.0, 2.0, 0.0, 0.0]), layout=alg.dense_layout())
    rhs = alg.multivector(jnp.array([0.0, 3.0, -4.0, 0.0]), layout=alg.dense_layout())

    try:
        actual = jax.jit(lambda a, b: (a * b).values)(lhs, rhs)
    finally:
        init(use="cpu")

    assert actual.shape == (4,)
    assert_allclose(np.asarray(actual), np.array([6.0, 3.0, -4.0, -8.0]), rtol=1e-5)


@pytest.mark.parametrize(
    "operation",
    [
        amsa.reverse,
        amsa.involute,
        amsa.conjugate,
        amsa.poincare_dual,
        amsa.poincare_undual,
    ],
)
def test_jax_jit_dense_unary_operations(operation):
    """Dense unary operations should trace through public functions."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga3d()
    mv = alg.multivector(
        {
            "e": jnp.array(1.0),
            "e1": jnp.array(2.0),
            "e23": jnp.array(-3.0),
            "e123": jnp.array(0.5),
        },
        layout=alg.dense_layout(),
    )

    try:
        actual = jax.jit(lambda value: operation(value).values)(mv)
    finally:
        init(use="cpu")

    expected = operation(
        alg.multivector(
            {"e": 1.0, "e1": 2.0, "e23": -3.0, "e123": 0.5},
            layout=alg.dense_layout(),
        )
    )
    assert_allclose(np.asarray(actual), np.asarray(expected.values), rtol=1e-5)


def test_jax_eval_shape_reports_dense_product_shape_without_execution():
    """Dense product shape should be available through jax.eval_shape."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga3d()
    lhs = alg.vector(jnp.ones((4, 3)))
    rhs = alg.vector(jnp.ones((4, 3)))

    try:
        shape = jax.eval_shape(lambda a, b: (a * b).values, lhs, rhs)
    finally:
        init(use="cpu")

    assert shape.shape == (4, 4)


def test_jax_jit_dense_grade_projection():
    """Dense-to-grade projection should trace when it only selects existing columns."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    register_backend("jax", JAXBackend())
    init(use="gpu")

    alg = amsa.Algebra.vga3d()
    mv = alg.multivector(
        jnp.arange(8.0),
        layout=alg.dense_layout(),
    )

    try:
        actual = jax.jit(lambda value: value.grade(1, 3).values)(mv)
    finally:
        init(use="cpu")

    assert_allclose(np.asarray(actual), np.array([1.0, 2.0, 4.0, 7.0]), rtol=1e-5)


def test_jax_jit_exp_coefficient_kernel_handles_all_signs():
    """Exp coefficient helper should trace without dynamic masking."""
    backend = JAXBackend()
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

    @jax.jit
    def coefficients(values):
        return backend.execute_sequence({"scalars": values}, ir)

    scalar, linear = coefficients(jnp.array([1.0, -1.0, 0.0]))

    assert_allclose(np.asarray(scalar), np.array([np.cosh(1.0), np.cos(1.0), 1.0]), rtol=1e-5)
    assert_allclose(np.asarray(linear), np.array([np.sinh(1.0), np.sin(1.0), 1.0]), rtol=1e-5)


def test_jax_jit_motor_exp_coefficient_kernel_handles_all_branches():
    """Motor exp coefficient helper should trace without Python value branches."""
    backend = JAXBackend()
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

    @jax.jit
    def coefficients(scalar, pseudoscalar):
        return backend.execute_sequence(
            {"scalar": scalar, "pseudoscalar": pseudoscalar},
            ir,
        )

    scalar, pseudo, linear, dual_linear = coefficients(
        jnp.array([0.0, -0.09, 0.16]),
        jnp.array([0.6, -0.12, 0.08]),
    )

    assert_allclose(np.asarray(scalar[0]), 1.0, rtol=1e-5)
    assert_allclose(np.asarray(pseudo[0]), 0.3, rtol=1e-5)
    assert_allclose(np.asarray(linear[0]), 1.0, rtol=1e-5)
    assert_allclose(np.asarray(dual_linear[0]), 0.1, rtol=1e-5)
    assert_allclose(np.asarray(scalar[1]), np.cos(0.3), rtol=1e-5)
    assert_allclose(np.asarray(linear[1]), np.sin(0.3) / 0.3, rtol=1e-5)
    assert_allclose(np.asarray(scalar[2]), np.cosh(0.4), rtol=1e-5)
    assert_allclose(np.asarray(linear[2]), np.sinh(0.4) / 0.4, rtol=1e-5)


def test_jax_jit_motor_log_coefficient_kernels_handle_zero_and_nonzero_cases():
    """Motor log coefficient helpers should trace without dynamic mask extraction."""
    backend = JAXBackend()
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

    @jax.jit
    def simple_coefficients(scalar, square):
        return backend.execute_sequence({"scalar": scalar, "square": square}, simple_ir)

    @jax.jit
    def pga3d_coefficients(scalar, pseudoscalar, sine):
        return backend.execute_sequence(
            {"scalar": scalar, "pseudoscalar": pseudoscalar, "sine": sine},
            pga3d_ir,
        )

    simple = simple_coefficients(jnp.array([1.0, 2.0]), jnp.array([0.0, -1.0]))
    alpha, beta = pga3d_coefficients(
        jnp.array([1.0, np.cos(0.3)]),
        jnp.array([0.0, -0.2 * np.sin(0.3)]),
        jnp.array([0.0, np.sin(0.3)]),
    )

    assert_allclose(np.asarray(simple), np.array([1.0, np.arctan2(1.0, 2.0)]), rtol=1e-5)
    assert_allclose(np.asarray(alpha[0]), 0.0, rtol=1e-5)
    assert_allclose(np.asarray(beta[0]), 0.0, rtol=1e-5)
    assert_allclose(np.asarray(alpha[1]), 0.3 / np.sin(0.3), rtol=1e-5)
    assert_allclose(
        np.asarray(beta[1]),
        0.2 * (1.0 - (0.3 * np.cos(0.3) / np.sin(0.3))) / np.sin(0.3),
        rtol=1e-5,
    )
