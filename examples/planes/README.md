# Planes Examples

This directory contains examples for representing planes and computing distances using multivectors.

## Examples

### [Point to Plane Distance 3D](point_plane_distance_3d.py)
Shows how to define a plane with two spanning vectors, generate its normal via the dual of the plane bivector, and calculate the signed distance from a point to that plane using Inner Product and components.

#### How to run:
```bash
uv run python examples/planes/point_plane_distance_3d.py
```

#### Expected Output:
```text
=== Point to Plane Distance ===
Plane bivector: [1. 0. 0.]
Plane normal: [ 0. -0.  1.]
Point: [0.5 0.5 2. ]

Signed distance to plane: 2.0
```
