# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA High-Performance Showcase: 1,000 Body Rigid Swarm (VisPy)
"""

import numpy as np

from amsa.algebra import Algebra
from amsa.viz.adapters import to_line_segments
from amsa.viz.backends import vispy as vback


def run_swarm(num_bodies=1000):
    print(f"\n=== AMSA High-Performance Swarm (N={num_bodies}) ===")

    # 1. Setup Algebra (PGA 2D)
    alg = Algebra.pga2d()

    # 2. Define Local Geometry (A single Triangle)
    local_vertices = alg.multivector(
        {"e12": [1.0, 1.0, 1.0], "e01": [0.2, -0.1, -0.1], "e02": [0.0, 0.2, -0.1]},
        batch_shape=(3,),
    )

    # 3. Initialize Random Motors
    motors = alg.multivector({"e12": 1.0}, batch_shape=(num_bodies, 1))

    vel_data = np.random.randn(num_bodies, 3) * 0.05
    velocities = alg.multivector(
        {"e12": vel_data[:, 0], "e01": vel_data[:, 1], "e02": vel_data[:, 2]},
        batch_shape=(num_bodies, 1),
    )

    dt = 0.5
    delta_motors = (velocities * (dt / 2.0)).exp()

    # 4. Setup Visualization with VisPy
    from vispy import scene

    canvas = scene.SceneCanvas(
        title=f"AMSA Swarm Showcase (N={num_bodies})", keys="interactive", show=True
    )
    view = canvas.central_widget.add_view()
    view.camera = "panzoom"
    view.camera.set_range(x=(-5, 5), y=(-5, 5))

    # Create initial line segments
    init_transformed = motors.sandwich(local_vertices)
    prim = to_line_segments(init_transformed, color="cyan")
    line = vback.plot(view.scene, prim, color="cyan")

    state = {"motors": motors}

    def update_frame(event):
        state["motors"] = delta_motors * state["motors"]
        state["motors"] = state["motors"].rigid_body_normalized()

        transformed = state["motors"].sandwich(local_vertices)
        prim = to_line_segments(transformed, color="cyan")
        line.set_data(pos=prim.positions)

        canvas.update()

    from vispy import app

    app.Timer(interval=1 / 60.0, connect=update_frame, start=True)

    print("Launching VisPy scene...")
    app.run()


if __name__ == "__main__":
    run_swarm(1000)
