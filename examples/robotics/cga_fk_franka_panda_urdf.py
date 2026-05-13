# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: CGA forward kinematics from a real-world URDF — Franka Panda
Algebra: 3D Conformal Geometric Algebra (CGA)

This example loads a flattened Panda URDF downloaded from a public robotics
dataset, converts it to AMSA's draft RobotModel shape, extracts the serial arm
chain from panda_link0 to panda_hand, and evaluates FK with CGA motors.

URDF source:
  https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/blob/main/
    robots/franka/urdf/franka_panda.urdf

The URDF contains hand/finger branches.  ``robo.serial_chain()`` selects the
arm path used for FK; branch-aware robot-graph execution is future work.
"""

from pathlib import Path

import numpy as np

import amsa.robo as robo
from amsa import Algebra


def _fmt_vec(values, width: int = 9) -> str:
    return ", ".join(f"{float(value):{width}.5f}" for value in values)


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "examples" / "robotics" / "assets" / "franka_panda.urdf"

print("\n=== CGA FK — Franka Panda from URDF ===\n")
print(f"URDF: {URDF.relative_to(ROOT)}")

alg = Algebra.cga3d()
model = robo.load(URDF, type="urdf")
arm = robo.serial_chain(model, "panda_link0", "panda_hand")

active = robo.active_joints(arm)
print(f"loaded model: {model.name}")
print("selected chain: panda_link0 -> panda_hand")
print(f"active joints: {', '.join(joint.name for joint in active)}")

# A typical non-singular Panda arm pose inside published joint limits.
q = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.785398163397])
print(f"\njoint values: ({_fmt_vec(q)})")

results = robo.fk_model(alg, arm, q)

for i, result in enumerate(results, start=1):
    joint = result["joint"]
    link = result["link"]
    position = result["position"]
    print(f"  {i:02d}. {joint:18s} -> {link:14s} pos ({_fmt_vec(position)})")

ee = results[-1]
print("\nend-effector pose:")
print(f"  link:        {ee['link']}")
print(f"  position:    ({_fmt_vec(ee['position'])})")
print(f"  orientation: ({_fmt_vec(ee['orientation'])})  quaternion (w, x, y, z)")

R = robo.motor_to_matrix(ee["motor"], alg)
print("\nrotation matrix columns recovered by CGA sandwich:")
for column in range(3):
    print(f"  | {_fmt_vec(R[:, column])} |")
