# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Robot trilateration
Algebra: 2D Projective Geometric Algebra (PGA)

Robots can estimate their position by measuring
distances to known landmark beacons.

This example demonstrates triangulation using
three beacon points.
"""

import matplotlib.pyplot as plt
import numpy as np

from amsa import Algebra
from amsa.viz.adapters import to_circle, to_point
from amsa.viz.backends import mpl

print("\n=== Robot Trilateration ===")

alg = Algebra.pga2d()

# beacon locations
b1 = alg.multivector({"e01": 0.0, "e02": 0.0, "e12": 1.0})
b2 = alg.multivector({"e01": 6.0, "e02": 0.0, "e12": 1.0})
b3 = alg.multivector({"e01": 3.0, "e02": 5.0, "e12": 1.0})

# robot location (unknown in real case)
robot = alg.multivector({"e01": 3.0, "e02": 2.0, "e12": 1.0})


def dist(p, q):
    dx = p.component("e01") - q.component("e01")
    dy = p.component("e02") - q.component("e02")
    return np.sqrt(dx * dx + dy * dy)


d1 = dist(robot, b1)
d2 = dist(robot, b2)
d3 = dist(robot, b3)

print("\nBeacon distances:")
print("b1:", round(d1, 3))
print("b2:", round(d2, 3))
print("b3:", round(d3, 3))

print("\nActual robot position:")
print(robot.component("e01"), robot.component("e02"))

# Visualization using the matplotlib backend
fig, ax = plt.subplots(figsize=(6, 6))

# Plot beacons and robot
mpl.plot(ax, to_point(b1, color="blue", label="Beacon 1"), size=100)
mpl.plot(ax, to_point(b2, color="blue", label="Beacon 2"), size=100)
mpl.plot(ax, to_point(b3, color="blue", label="Beacon 3"), size=100)
mpl.plot(ax, to_point(robot, color="red", label="Robot"), size=100)

# Plot distance circles
for b, r in [(b1, d1), (b2, d2), (b3, d3)]:
    pt = to_point(b)
    mpl.plot(
        ax,
        to_circle(pt.position, r, color="blue", label=None),
        linestyle="--",
        alpha=0.5,
    )

ax.set_aspect("equal", "box")
ax.set_title("Robot Trilateration Example")
ax.grid(True)
ax.legend()

print("\nDisplaying visualization plot...")
mpl.show()
