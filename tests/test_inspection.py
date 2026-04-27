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

"""Tests for inspection and pretty-print API."""

import numpy as np
import pytest

from amsa import Algebra


def test_mvarray_repr_simple():
    """Test MVArray.__repr__ for simple multivector."""
    alg = Algebra.vga2d()
    mv = alg.vector([1.0, 2.0])
    repr_str = repr(mv)
    assert "e1" in repr_str or "e2" in repr_str
    assert "1.0" in repr_str or "2.0" in repr_str


def test_mvarray_repr_zero():
    """Test MVArray.__repr__ for zero multivector."""
    alg = Algebra.vga2d()
    mv = alg.zeros()
    repr_str = repr(mv)
    assert repr_str == "0"


def test_mvarray_repr_bivector():
    """Test MVArray.__repr__ for bivector."""
    alg = Algebra.vga2d()
    mv = alg.bivector([3.0])
    repr_str = repr(mv)
    assert "e12" in repr_str
    assert "3.0" in repr_str


def test_mvarray_repr_batched():
    """Test MVArray.__repr__ for batched multivector."""
    alg = Algebra.vga2d()
    mv = alg.zeros(batch_shape=(2, 3))
    repr_str = repr(mv)
    assert "batch_shape=(2, 3)" in repr_str
    assert "blades=" in repr_str
    assert "dtype=" in repr_str


def test_mvarray_repr_mixed_grades():
    """Test MVArray.__repr__ for mixed-grade multivector."""
    alg = Algebra.vga2d()
    mv = alg.multivector({0: 1.0, 1: 2.0, 3: 3.0})
    repr_str = repr(mv)
    assert "e" in repr_str or "e1" in repr_str or "e12" in repr_str
