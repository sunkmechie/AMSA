# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA Real-time Example: 2D Rigid Body Swarm

This example demonstrates the power of AMSA's batch execution and PGA.
It simulates N rigid bodies (triangles) moving in the plane.

Mathematical approach:
1. Each body has a local vertex set (local bivectors).
2. Each body has a pose stored as a Motor (rotor + translator).
3. Incremental updates are performed by multiplying the motor by a 'delta' motor.
4. Vertices are transformed to world space using the sandwich product: X' = M X M^-1.
"""

import numpy as np

from amsa import Algebra
from amsa import viz

def run_simulation(num_bodies=50):
    print(f"\n=== AMSA Real-time Rigid Body Swarm (N={num_bodies}) ===")
    
    alg = Algebra.pga2d()
    
    # 1. Define local vertices of a triangle (centroid at origin)
    # In PGA2D, a point is a bivector: x*e01 + y*e02 + 1*e12
    # We create a batch of 3 points (the triangle vertices)
    local_data = {
        "e01": np.array([[-0.1,  0.1,  0.0]]),
        "e02": np.array([[-0.1, -0.1,  0.1]]),
        "e12": np.array([[ 1.0,  1.0,  1.0]]),
    }
    local_vertices = alg.multivector(local_data) # Shape: (3,)
    
    # 2. Initialize N bodies with random positions and orientations
    # We use a batched MVArray of shape (N,)
    rng = np.random.default_rng(42)
    
    initial_x = rng.uniform(-4, 4, num_bodies)[:, np.newaxis]
    initial_y = rng.uniform(-4, 4, num_bodies)[:, np.newaxis]
    initial_angle = rng.uniform(0, 2*np.pi, num_bodies)[:, np.newaxis]
    
    # Rotation (Rotor)
    rotors = alg.multivector({
        "e": np.cos(initial_angle / 2),
        "e12": -np.sin(initial_angle / 2)
    })
    
    # Translation (Translator)
    translators = alg.multivector({
        "e": 1.0,
        "e01": -0.5 * initial_y,
        "e02": 0.5 * initial_x
    })
    
    # Initial motors
    motors = translators * rotors
    
    # 3. Define incremental "velocity" motors
    # Each body rotates at a slightly different speed and moves in a direction
    twist_speeds = rng.uniform(0.01, 0.05, num_bodies)[:, np.newaxis]
    drift_x = rng.uniform(-0.02, 0.02, num_bodies)[:, np.newaxis]
    drift_y = rng.uniform(-0.02, 0.02, num_bodies)[:, np.newaxis]
    
    # Delta motor components
    # Small angle rotation + small translation
    d_rotors = alg.multivector({
        "e": np.cos(twist_speeds / 2),
        "e12": -np.sin(twist_speeds / 2)
    })
    d_translators = alg.multivector({
        "e": 1.0,
        "e01": -0.5 * drift_y,
        "e02": 0.5 * drift_x
    })
    delta_motors = d_rotors * d_translators
    
    # 4. Setup Visualization
    # viz.view() automatically creates the figure and axes (or VisPy scene)
    # We force backend="mpl" here because this script uses FuncAnimation
    ax = viz.view(motors, backend="mpl", title="AMSA 2D Swarm")
    
    # We create the line objects (artists) for each body
    # Since viz.view returned a matplotlib Axes, we can use it
    lines = []
    for i in range(num_bodies):
        ln, = ax.plot([], [], 'o-', lw=2, markersize=4)
        lines.append(ln)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    state = {"motors": motors}
    
    def update(frame):
        # A. Update motors: M = dM * M
        state["motors"] = delta_motors * state["motors"]
        
        # B. Rigidly normalize to prevent drift over time
        state["motors"] = state["motors"].rigid_body_normalized()
        
        # C. Transform vertices: X' = M X M^-1
        transformed = state["motors"].sandwich(local_vertices)
        
        # D. Extract coordinates using viz adapter
        pos = viz.to_point(transformed).position
        
        # E. Update artists
        for i in range(num_bodies):
            tx = [*pos[i, :, 0], pos[i, 0, 0]]
            ty = [*pos[i, :, 1], pos[i, 0, 1]]
            lines[i].set_data(tx, ty)
        
        return lines

    print("Starting animation...")
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(ax.figure, update, frames=200, interval=20, blit=True)
    viz.show()

if __name__ == "__main__":
    run_simulation(60)
