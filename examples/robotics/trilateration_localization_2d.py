# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Example

Topic: Robot trilateration
Algebra: 2D Projective Geometric Algebra (PGA)

Robots can estimate their position by measuring
distances to known landmarks (beacons).

This example demonstrates triangulation using
three beacon points.
"""
import numpy as np

from amsa import Algebra

print("\n=== Robot Trilateration ===")

alg = Algebra.pga2d()

# --------------------------------------------------
# beacon locations
# --------------------------------------------------

b1 = alg.multivector({"e01": 0.0, "e02": 0.0, "e12": 1.0})
b2 = alg.multivector({"e01": 6.0, "e02": 0.0, "e12": 1.0})
b3 = alg.multivector({"e01": 3.0, "e02": 5.0, "e12": 1.0})

# robot location (unknown in real case)
robot = alg.multivector({"e01": 3.0, "e02": 2.0, "e12": 1.0})

# compute distances (simulated sensor readings)



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

try:
    import matplotlib.pyplot as plt

    from amsa.viz.adapters import to_point
    from amsa.viz.backends import mpl

    fig, ax = plt.subplots(figsize=(6, 6))

    mpl.plot(ax, to_point(b1, color="blue", label="Beacon 1"))
    mpl.plot(ax, to_point(b2, color="blue", label="Beacon 2"))
    mpl.plot(ax, to_point(b3, color="blue", label="Beacon 3"))
    mpl.plot(ax, to_point(robot, color="red", label="Robot"))

    # Plot distance circles
    for b, r in [(b1, d1), (b2, d2), (b3, d3)]:
        pt = to_point(b)
        circle = plt.Circle(pt.position, r, color="blue", fill=False, linestyle="--", alpha=0.5)
        ax.add_patch(circle)

    ax.set_aspect("equal", "box")
    ax.set_title("Robot Trilateration Example")
    ax.grid(True)
    ax.legend()
    
    print("\nDisplaying visualization plot...")
    mpl.show()
except ImportError:
    print("\nSkipping visualization... matplotlib or amsa.viz is not available.")
