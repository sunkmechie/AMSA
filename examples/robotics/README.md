# Robotics Examples

This directory contains examples detailing how AMSA can be applied to common problems in robotics, including motion planning, geometry processing, and localization.

## Examples

### [Corridor Corner Detection (PGA 2D)](pga_corridor_corner_2d.py)
Demonstrates how to find the intersection of two lines (representing walls) using the regressive product in PGA. This is a common task in processing sensor data to find environment features like corners.

### [Rigid Body Trajectory (PGA 2D)](pga_rigid_body_trajectory_2d.py)
Simulates a rigid body's trajectory by repeatedly applying a motor (composed of translation and rotation) to a point, demonstrating an alternative to homogeneous transformation matrices.

### [Planar Heading Update (VGA 2D)](planar_heading_rotor.py)
Uses a rotor in 2D Vector Geometric Algebra (VGA) to rotate a robot's forward-facing axis, showing how simple rotations are handled in GA.

### [Ray-Plane Reflection (VGA 3D)](ray_plane_reflection_3d.py)
Demonstrates reflecting a velocity vector across various coordinate planes using the sandwich product with a plane normal.

### [Robot Rotation Comparison (VGA 2D)](robot_rotation_comparison_2d.py)
Provides a direct comparison between rotation matrices, quaternions, and GA rotors for a 45-degree rotation, verifying that they yield identical results.

### [Trilateration Localization (PGA 2D)](trilateration_localization_2d.py)
Simulates a robot's position estimation by calculating the distances to three known landmark beacons. Uses `amsa.viz` with the matplotlib backend to render the beacons, robot estimate, and distance circles.

### [Vector Projection (VGA 2D)](vector_projection_2d.py)
Projects a robot's position onto a corridor axis using the inner product, a common task in navigation and alignment.

### [3D Kinematic Chain (PGA 3D)](pga3d_kinematic_chain.py)
Animated 3D robot arm using PGA3D motors and VisPy. Demonstrates forward kinematics with joint motors, coordinate frame rendering, and real-time animation.

### [CGA FK — 3-Link Non-Planar Arm](cga_fk_3link.py)
Computes forward kinematics for a 3-DOF arm with a twisted joint (α = π/3) using Denavit–Hartenberg motor composition. Shows that the CGA formulation handles non-planar kinematics without matrices.

```bash
uv run python examples/robotics/cga_fk_3link.py
```

### [CGA FK — UR5 6-DOF Industrial Arm](cga_fk_ur5.py)
Computes forward kinematics for a Universal Robots UR5 using the official DH parameters from UR documentation. Demonstrates CGA FK on a real industrial 6-DOF manipulator.

```bash
uv run python examples/robotics/cga_fk_ur5.py
```

### [CGA FK — Franka Panda from URDF](cga_fk_franka_panda_urdf.py)
Loads a real flattened Franka Panda URDF from `examples/robotics/assets/`, extracts the serial arm chain, and evaluates FK through CGA motors.

```bash
uv run python examples/robotics/cga_fk_franka_panda_urdf.py
```

### [CGA IK — UR5 6-DOF Industrial Arm (DLS)](cga_ik_ur5.py)
Solves inverse kinematics for the UR5 using damped least-squares (Levenberg-Marquardt) numerical IK. Demonstrates motor-space target specification, geometric Jacobian construction, adaptive damping, joint-limit enforcement, and self-consistency round-trip verification.

```bash
uv run python examples/robotics/cga_ik_ur5.py
```

### [CGA IK — UR5 Solver Comparison](cga_ik_ur5_solver_comparison.py)
Compares the full-chain UR5 DLS IK solver with the newer CGA geometric IK primitive solvers. DLS solves the complete end-effector motor target, while the CGA sphere-sphere and point-circle solver paths recover elbow geometry from the same UR5 target chain.

```bash
uv run python examples/robotics/cga_ik_ur5_solver_comparison.py
```

## How to run
You can run any of these examples using `uv`:

```bash
uv run python examples/robotics/[example_name].py
```

The 2D visualization example expects the `viz` extra so matplotlib is available, and the 3D showcase expects the same extra for VisPy.

## Sample Output (Rigid Body Trajectory)
```text
=== Rigid Body Trajectory (PGA Motor) ===

Robot trajectory:
step 01 -> (0.500, 0.000)
step 02 -> (0.992, 0.087)
step 03 -> (1.462, 0.258)
...
step 20 -> (4.908, 4.382)
```
