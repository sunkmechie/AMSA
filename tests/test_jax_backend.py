import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import amsa  # noqa: E402
from amsa.backends.jax import JAXBackend  # noqa: E402
from amsa.backends.numpy import NumpyBackend  # noqa: E402
from amsa.ir import clear_backends, init, register_backend  # noqa: E402
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
