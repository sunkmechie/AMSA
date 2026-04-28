import pytest

import amsa
from amsa.backends.numpy import NumpyBackend
from amsa.ir import clear_backends, get_device, init, register_backend


def test_init_cpu():
    """Test initializing with CPU device."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    assert get_device() == "cpu"


def test_init_invalid_device():
    """Test that invalid device raises ValueError."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    with pytest.raises(ValueError, match="Unsupported device"):
        init(use="tpu")


def test_init_unregistered_backend():
    """Test that device with unregistered backend raises ValueError."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    with pytest.raises(ValueError, match="Backend.*not available"):
        init(use="gpu")


def test_get_device():
    """Test getting current device."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    assert get_device() == "cpu"


def test_operations_use_selected_backend():
    """Test that operations work after backend selection."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 2.0])
    v = alg.vector([3.0, -4.0])
    
    result = u * v
    assert result is not None
    # Geometric product of two vectors in VGA2D returns scalar + bivector (2 components)
    assert result.values.shape == (2,)


def test_algebra_semantics_unchanged():
    """Test that backend selection does not affect algebra semantics."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    
    alg = amsa.Algebra.vga2d()
    u = alg.vector([1.0, 0.0])
    v = alg.vector([0.0, 1.0])
    
    # Outer product of orthogonal vectors should be bivector
    op = u ^ v
    assert op.layout.blades == (3,)  # e12 in VGA2D
    
    # Inner product should be scalar
    ip = u | v
    assert ip.layout.blades == (0,)  # scalar


def test_backend_persistence():
    """Test that backend selection persists across operations."""
    clear_backends()
    register_backend("numpy", NumpyBackend())
    init(use="cpu")
    
    assert get_device() == "cpu"
    
    # Perform multiple operations
    alg = amsa.Algebra.vga2d()
    for _ in range(3):
        u = alg.vector([1.0, 2.0])
        v = alg.vector([3.0, -4.0])
        _ = u * v
    
    # Device should still be cpu
    assert get_device() == "cpu"


def test_public_api_exports():
    """Test that init and get_device are exported in public API."""
    assert hasattr(amsa, "init")
    assert hasattr(amsa, "get_device")
    assert callable(amsa.init)
    assert callable(amsa.get_device)
