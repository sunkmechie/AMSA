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

from amsa.algebra import Algebra, EntityInfo

# Register the NumPy IR backend as the default execution engine.
from amsa.backends.numpy import NumpyBackend
from amsa.ir import get_device, init, register_backend
from amsa.layouts import MVLayout
from amsa.mv import MVArray
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
    divide,
    dual,
    exp,
    geometric_product,
    inner_product,
    inverse,
    involute,
    left_contraction,
    log,
    motor_exp,
    motor_log,
    neg,
    norm,
    norm_squared,
    normalize,
    outer_product,
    poincare_dual,
    poincare_undual,
    project_grades,
    regressive_product,
    reverse,
    right_contraction,
    rigid_body_normalize,
    sandwich,
    scalar_product,
    sub,
    undual,
    unitize,
    weight,
    weight_dual,
    weight_norm,
    weight_norm_squared,
)
from amsa.specs import (
    AlgebraSpec,
    cga2d,
    cga3d,
    grade_of_blade,
    pga2d,
    pga3d,
    vga,
    vga2d,
    vga3d,
)

register_backend("numpy", NumpyBackend())

# Register JAX backend if available
try:
    from amsa.backends.jax import JAXBackend
    register_backend("jax", JAXBackend())
except ImportError:
    pass  # JAX not installed

init(use="cpu")

__all__ = [
    "Algebra",
    "AlgebraSpec",
    "EntityInfo",
    "MVArray",
    "MVLayout",
    "get_device",
    "init",
    "add",
    "anticommutator_product",
    "bulk",
    "bulk_dual",
    "bulk_norm",
    "bulk_norm_squared",
    "bulk_normalize",
    "commutator_product",
    "conjugate",
    "divide",
    "dual",
    "exp",
    "geometric_product",
    "grade_of_blade",
    "inner_product",
    "inverse",
    "involute",
    "left_contraction",
    "log",
    "motor_exp",
    "motor_log",
    "neg",
    "norm",
    "norm_squared",
    "normalize",
    "outer_product",
    "pga2d",
    "pga3d",
    "cga2d",
    "cga3d",
    "poincare_dual",
    "poincare_undual",
    "project_grades",
    "regressive_product",
    "reverse",
    "rigid_body_normalize",
    "right_contraction",
    "sandwich",
    "scalar_product",
    "sub",
    "undual",
    "unitize",
    "vga",
    "vga2d",
    "vga3d",
    "weight",
    "weight_dual",
    "weight_norm",
    "weight_norm_squared",
]
