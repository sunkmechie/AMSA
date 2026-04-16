# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA High-Performance Showcase: 1,000 Body Rigid Swarm (VisPy + GLFW)
"""

import numpy as np

from amsa import viz
from amsa.algebra import Algebra


def run_swarm(num_bodies=1000):
    print(f"\n=== AMSA High-Performance Swarm (N={num_bodies}) ===")
    
    # 1. Setup Algebra (PGA 2D)
    alg = Algebra.pga2d()
    
    # 2. Define Local Geometry (A single Triangle)
    # Points in PGA2D are bivectors: e12 (origin), e01, e02
    local_vertices = alg.multivector({
        "e12": [1.0, 1.0, 1.0], 
        "e01": [0.2, -0.1, -0.1], 
        "e02": [0.0, 0.2, -0.1]
    }, batch_shape=(3,))
    
    # 3. Initialize Random Motors (Identity to start)
    motors = alg.multivector({"e12": 1.0}, batch_shape=(num_bodies, 1))
    
    # Random initial velocities (bivectors)
    # Rotation (e12) + Translation (e01, e02)
    vel_data = np.random.randn(num_bodies, 3) * 0.05
    velocities = alg.multivector({
        "e12": vel_data[:, 0],
        "e01": vel_data[:, 1],
        "e02": vel_data[:, 2]
    }, batch_shape=(num_bodies, 1))
    
    # Differential motors for the update step: dM = exp(v * dt / 2)
    dt = 0.5
    delta_motors = (velocities * (dt / 2.0)).exp()
    
    # 4. Setup Visualization
    from amsa.viz.backends.vispy import AMSAScene
    
    # We use our new vectorized adapter for the initial state
    init_transformed = motors.sandwich(local_vertices)
    swarm_prim = viz.to_line_segments(init_transformed, color="cyan")
    
    # Initialize the scene
    scene = AMSAScene(title=f"AMSA Swarm Showcase (N={num_bodies})")
    layer = viz.Layer(
        artist=scene.add(swarm_prim, width=1.5),
        primitive=viz.LineSegments,
        backend="vispy",
        parent=scene.view
    )
    
    # Center the camera
    scene.view.camera = "panzoom"
    scene.view.camera.set_range(x=(-5, 5), y=(-5, 5))
    
    state = {"motors": motors}
    
    def update_frame(event):
        # A. Update motors: M = dM * M
        state["motors"] = delta_motors * state["motors"]
        
        # B. Rigidly normalize
        state["motors"] = state["motors"].rigid_body_normalized()
        
        # C. Transform vertices (Vectorized)
        transformed = state["motors"].sandwich(local_vertices)
        
        # D. High-speed Direct update
        viz.update(layer, transformed)
        
        # E. Request canvas update
        scene.canvas.update()

    # Create a high-speed timer (60 FPS)
    from vispy import app
    timer = app.Timer(interval=1/60.0, connect=update_frame, start=True)
    
    print("Launching VisPy scene...")
    scene.show()

if __name__ == "__main__":
    # We use 1000 for the demo, can be cranked higher!
    run_swarm(1000)
