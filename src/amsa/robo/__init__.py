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

"""Experimental robotics subpackage for AMSA.

This subpackage provides robot model types, URDF/crobot I/O, DH-based
forward and inverse kinematics, motor extraction utilities, and CGA
intersection helpers.  It will eventually be split into the standalone
``amsa-robo`` package.
"""

from amsa.robo.geometry import line_plane, point_circle_projection, sphere_sphere
from amsa.robo.kinematics import (
    IKResult,
    _fk_frames,  # noqa: F401
    fk,
    fk_model,
    ik,
    ik_cga_spherical_wrist,
    ik_dls,
    ik_model_dls,
    joint_motion_motor,
    motor_to_matrix,
    motor_to_position,
    motor_to_quaternion,
    planar_two_link_ik,
)

# --- temporary visibility for tests that reference private helpers ---
from amsa.robo.model import (  # noqa: F401
    Joint,
    Link,
    RobotModel,
    _default_joint_motion,
    _resolve_model_joint_values,
    active_joints,
    dump_crobot,
    importurdf,
    load,
    load_crobot,
    model_from_dh,
    serial_chain,
)

EXPERIMENTAL_WARNING = (
    "amsa.robo is experimental and not ready for production robotics use. "
    "APIs and file formats may change before amsa-robo is split out."
)

__all__ = [
    "EXPERIMENTAL_WARNING",
    "IKResult",
    "Joint",
    "Link",
    "RobotModel",
    "active_joints",
    "dump_crobot",
    "fk",
    "fk_model",
    "ik",
    "ik_cga_spherical_wrist",
    "ik_dls",
    "ik_model_dls",
    "importurdf",
    "joint_motion_motor",
    "line_plane",
    "load",
    "load_crobot",
    "model_from_dh",
    "motor_to_matrix",
    "motor_to_position",
    "motor_to_quaternion",
    "planar_two_link_ik",
    "point_circle_projection",
    "serial_chain",
    "sphere_sphere",
]
