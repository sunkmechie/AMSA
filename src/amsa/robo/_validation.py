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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amsa.algebra import Algebra
    from amsa.mv import MVArray


def _validate_cga3d(alg: Algebra) -> None:
    if alg.dimension != 5 or alg.signature != (1, 1, 1, 1, -1):
        raise ValueError("Experimental robotics CGA helpers require Algebra.cga3d().")


def _validate_motor_algebra(motor: MVArray, alg: Algebra) -> None:
    if motor.algebra != alg.spec:
        raise ValueError("Motor must belong to the provided algebra.")


def _validate_same_cga(*values: MVArray) -> None:
    if not values:
        return
    algebra = values[0].algebra
    from amsa.algebra import Algebra

    alg = Algebra(algebra)
    _validate_cga3d(alg)
    for value in values:
        if value.algebra != algebra:
            raise ValueError("CGA objects must belong to the same algebra.")


def _validate_joint_types(n: int, joint_types: list[str] | None) -> list[str]:
    resolved = ["revolute"] * n if joint_types is None else list(joint_types)
    if len(resolved) != n:
        raise ValueError(f"Expected {n} joint types, got {len(resolved)}.")
    invalid = sorted(set(resolved) - {"revolute", "prismatic"})
    if invalid:
        names = ", ".join(repr(item) for item in invalid)
        raise ValueError(f"Unsupported joint type(s): {names}.")
    return resolved
