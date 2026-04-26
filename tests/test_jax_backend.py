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

import pytest

from amsa.ir import has_backend


def test_jax_backend_registration():
    """Test JAX backend registration behavior."""
    try:
        import jax  # noqa: F401
        # If JAX is available, backend should be registered
        assert has_backend("jax")
    except ImportError:
        # If JAX is not available, backend should not be registered
        assert not has_backend("jax")
