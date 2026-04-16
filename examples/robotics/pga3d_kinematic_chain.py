# Copyright 2026 Surya Sunkara
# SPDX-License-Identifier: Apache-2.0

"""
AMSA 3D Showcase: PGA3D Kinematic Chain
"""

import numpy as np

import amsa
from amsa import viz
from amsa.algebra import Algebra


def run_3d_arm():
    print("\n=== AMSA 3D Robot Arm (PGA3D) ===")
    
    # 1. Setup Algebra (PGA 3D)
    alg = Algebra.pga3d()
    
    # 2. Define Local Linkage points
    # Points in PGA3D: P = w*e123 - x*e023 + y*e013 - z*e012
    p_base = alg.multivector({"e123": 1.0})
    p_elbow_local = alg.multivector({"e123": 1.0, "e012": -2.0}) # 2 units in Z
    p_wrist_local = alg.multivector({"e123": 1.0, "e012": -2.0, "e013": 2.0}) # +2 in Y
    
    # 3. Setup Visualization
    # Use viz.view() to start the scene with the arm segments
    # This automatically selects the best backend (VisPy for 3D)
    arm_pts = alg.multivector(np.stack([
        p_base.as_dense().values, 
        p_elbow_local.as_dense().values, 
        p_wrist_local.as_dense().values
    ]))
    arm_primitive = viz.to_line_segments(arm_pts, color="cyan", connect="strip")
    arm_layer = viz.view(arm_primitive, title="AMSA 3D Robot Arm", width=5, backend="vispy")
    
    # Get the view/axes for subsequent plotting
    view = arm_layer.parent
    
    # Add joint frames and guides via viz.plot() facade
    # The facade returns Layer objects which are used for updates
    frame1_layer = viz.plot(viz.to_rotor(alg.multivector({"e": 1.0})), parent=view)
    frame2_layer = viz.plot(viz.to_rotor(alg.multivector({"e": 1.0})), parent=view)
    
    # Joint Axes (Visual guides)
    axis1 = alg.multivector({"e12": 1.0}) # Z
    axis2_local = alg.multivector({"e13": -1.0}) # Y (e31)
    axis1_layer = viz.plot(viz.to_line(axis1, color="red"), parent=view)
    axis2_layer = viz.plot(viz.to_line(axis1, color="green"), parent=view) # Will be updated
    
    # 4. Animation logic
    state = {"time": 0.0}
    
    def update(event):
        t = state["time"]
        
        # --- Kinematics ---
        # Joint 1: Base rotation (Z-axis)
        m1 = (axis1 * (t * 0.5 / 2.0)).exp()
        
        # Joint 2: Shoulder rotation (Y-axis, relative to m1)
        elbow_trans = alg.multivector({"e": 1.0, "e03": -1.0}) 
        axis2_world = m1.sandwich(elbow_trans.sandwich(axis2_local))
        
        m2_local = (axis2_world * (np.sin(t) * 1.0 / 2.0)).exp()
        m2_total = m2_local * m1
        
        # Calculate world-space points
        base = p_base
        elbow = m1.sandwich(p_elbow_local)
        wrist = m2_total.sandwich(p_wrist_local)
        
        # Combine points into a batch for update
        arm_pts = alg.multivector(np.stack([
            base.as_dense().values, 
            elbow.as_dense().values, 
            wrist.as_dense().values
        ]))
        
        # --- Update Visuals via amsa.vizFacade ---
        viz.update(arm_layer, arm_pts)
        viz.update(frame1_layer, m1)
        viz.update(frame2_layer, m2_total)
        viz.update(axis2_layer, axis2_world)
        
        state["time"] += 0.03

    # 5. Start Animation Loop
    if hasattr(view, "camera"): # VisPy Camera setup
        view.camera = "turntable"
        view.camera.distance = 15
        
        from vispy import app
        timer = app.Timer(interval=1/60.0, connect=update, start=True)
        
    print("Starting animation...")
    viz.show()

if __name__ == "__main__":
    run_3d_arm()
