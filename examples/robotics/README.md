# Robotics Examples

This directory contains examples detailing how AMSA can be applied to common problems in robotics, including motion planning, geometry processing, and localization.

## Examples

### [Circular Robot Motion (PGA 2D)](pga_circle_motion_2d.py)
Uses 2D Projective Geometric Algebra (PGA) to represent and simulate the motion of a differential-drive robot moving along a circular path. It uses motors (translation + rotation) to evolve the robot's pose.

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
Simulates a robot's position estimation by calculating the distances to three known landmark beacons.

### [Vector Projection (VGA 2D)](vector_projection_2d.py)
Projects a robot's position onto a corridor axis using the inner product, a common task in navigation and alignment.

## How to run
You can run any of these examples using `uv`:

```bash
uv run python examples/robotics/[example_name].py
```

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
