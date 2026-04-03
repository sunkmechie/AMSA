# Geometry Examples

This directory contains examples of geometric computations using Clifford Algebra.

## Examples

### [Orientation Batch 2D](orientation_batch_2d.py)
Demonstrates batched orientation testing for many triangles using the wedge product. The sign of the bivector component determines if a triangle is counter-clockwise, clockwise, or degenerate.

#### How to run:
```bash
uv run python examples/geometry/orientation_batch_2d.py
```

### [Signed Volume 3D](signed_volume_3d.py)
Computes the signed volume of a parallelepiped using the outer product of three vectors in 3D VGA.

#### How to run:
```bash
uv run python examples/geometry/signed_volume_3d.py
```

#### Expected Output:
```text
=== Signed Volume Example ===
u: [1. 0. 0.]
v: [0. 2. 0.]
w: [0. 0. 3.]

Trivector: [6.]
Signed volume: 6.0
```

### [Triangle Area 2D](triangle_area_2d.py)
Uses the wedge product to calculate the oriented area of a triangle in the plane. Demonstrates antisymmetry and translation invariance.

#### How to run:
```bash
uv run python examples/geometry/triangle_area_2d.py
```

#### Expected Output:
```text
--- Right triangle example ---
Area is 7.5

--- Orientation flip ---
Flipped area is -7.5

 --- Skew triangle example ---
Area is 6.0
```
