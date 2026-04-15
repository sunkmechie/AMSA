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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from amsa import Algebra
from amsa.viz import to_point

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
    
    # 4. Setup Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title(f"AMSA Real-time Swarm: {num_bodies} Rigid Bodies in PGA")
    
    # Pre-create lines for each triangle
    lines = [ax.plot([], [], 'b-', lw=1)[0] for _ in range(num_bodies)]
    
    state = {"motors": motors}
    
    def update(frame):
        # A. Update motors: M = dM * M
        state["motors"] = delta_motors * state["motors"]
        
        # B. Rigidly normalize to prevent drift over time
        state["motors"] = state["motors"].rigid_body_normalized()
        
        # C. Transform vertices: X' = M X M^-1
        # motors has shape (N, 1), local_vertices has shape (3,) -> (N, 3)
        transformed = state["motors"].sandwich(local_vertices)
        
        # D. Extract coordinates using amsa.viz adapter
        # Point.position will have shape (N, 3, 2)
        pos = to_point(transformed).position
        
        # E. Update artists
        for i in range(num_bodies):
            # Triangle vertices (close the loop)
            tx = [*pos[i, :, 0], pos[i, 0, 0]]
            ty = [*pos[i, :, 1], pos[i, 0, 1]]
            lines[i].set_data(tx, ty)
        
        return lines

    print("Starting animation...")
    ani = FuncAnimation(fig, update, frames=200, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation(60)
