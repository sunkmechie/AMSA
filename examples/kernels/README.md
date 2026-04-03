# Kernels Examples

This directory contains examples showing how common geometric computations can be written directly using multivector operations.

## Examples

### [Geometric Kernels](geometric_kernels.py)
A collection of mini-geometry utilities like:
- `signed_area_2d(a, b, c)`
- `signed_volume_3d(u, v, w)`
- `are_orthogonal(u, v)`
- `bivector_plane(u, v)`

Shows how to replace matrix-based operations with multivector formulas for better clarity and efficiency.

#### How to run:
```bash
uv run python examples/kernels/geometric_kernels.py
```

#### Expected Output:
```text
=== Mini Geometry Kernel ===
Triangle area: 6.0
Signed volume: 1.0
Are u and v orthogonal? True
```
