# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import amsa  # noqa: E402
from amsa.backends.jax import JAXBackend  # noqa: E402
from amsa.backends.numpy import NumpyBackend  # noqa: E402
from amsa.ir import clear_backends, init, register_backend  # noqa: E402
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
