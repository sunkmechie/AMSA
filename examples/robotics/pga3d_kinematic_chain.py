# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA 3D Showcase: PGA3D Kinematic Chain
"""

import numpy as np

from amsa.algebra import Algebra
from amsa.viz.adapters import to_line, to_line_segments, to_rotor
from amsa.viz.backends import vispy as vback


def run_3d_arm():
    print("\n=== AMSA 3D Robot Arm (PGA3D) ===")

    # 1. Setup Algebra (PGA 3D)
    alg = Algebra.pga3d()

    # 2. Define Local Linkage points
    p_base = alg.multivector({"e123": 1.0})
    p_elbow_local = alg.multivector({"e123": 1.0, "e012": -2.0})
    p_wrist_local = alg.multivector({"e123": 1.0, "e012": -2.0, "e013": 2.0})

    # 3. Setup Visualization with VisPy
    from vispy import scene

    canvas = scene.SceneCanvas(title="AMSA 3D Robot Arm", keys="interactive", show=True)
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    view.camera.distance = 15

    # Create arm line segments
    arm_pts = alg.multivector(
        np.stack(
            [
                p_base.as_dense().values,
                p_elbow_local.as_dense().values,
                p_wrist_local.as_dense().values,
            ]
        )
    )
    arm_primitive = to_line_segments(arm_pts, color="cyan", connect="strip")
    arm_line = vback.plot(view.scene, arm_primitive, color="cyan")

    # Add joint frames
    frame1 = vback.plot(view.scene, to_rotor(alg.multivector({"e": 1.0})))

    frame2 = vback.plot(view.scene, to_rotor(alg.multivector({"e": 1.0})))

    # Add axes
    axis1 = to_line(alg.multivector({"e12": 1.0}), color="red")
    vback.plot(view.scene, axis1, color="red", scale=1000)

    axis2_line = vback.plot(view.scene, axis1, color="green", scale=1000)

    # 4. Animation
    state = {"time": 0.0}

    def update(event):
        t = state["time"]

        m1 = (alg.multivector({"e12": 1.0}) * (t * 0.5 / 2.0)).exp()

        elbow_trans = alg.multivector({"e": 1.0, "e03": -1.0})
        axis2_local = alg.multivector({"e13": -1.0})
        axis2_world = m1.sandwich(elbow_trans.sandwich(axis2_local))

        m2_local = (axis2_world * (np.sin(t) * 1.0 / 2.0)).exp()
        m2_total = m2_local * m1

        base = p_base
        elbow = m1.sandwich(p_elbow_local)
        wrist = m2_total.sandwich(p_wrist_local)

        arm_pts = alg.multivector(
            np.stack([base.as_dense().values, elbow.as_dense().values, wrist.as_dense().values])
        )
        arm_primitive = to_line_segments(arm_pts, color="cyan", connect="strip")
        arm_line.set_data(pos=arm_primitive.positions)

        # Update frames
        r1 = to_rotor(m1)
        mat1 = r1.matrix.T
        mat1_4 = np.eye(4)
        mat1_4[:3, :3] = mat1
        frame1.transform.matrix = mat1_4

        r2 = to_rotor(m2_total)
        mat2 = r2.matrix.T
        mat2_4 = np.eye(4)
        mat2_4[:3, :3] = mat2
        frame2.transform.matrix = mat2_4

        # Update axis2
        axis2 = to_line(axis2_world, color="green")
        p2 = axis2.origin
        d2 = axis2.direction / np.linalg.norm(axis2.direction)
        axis2_line.set_data(pos=np.stack([p2 - d2 * 1000, p2 + d2 * 1000]))

        state["time"] += 0.03

    from vispy import app

    app.Timer(interval=1 / 60.0, connect=update, start=True)

    print("Starting animation...")
    app.run()


if __name__ == "__main__":
    run_3d_arm()
