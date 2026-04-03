# Algebra Examples

This directory contains examples demonstrating core algebraic properties and operations in AMSA.

## Examples

### [Even / Odd Decomposition](even_odd_decomposition.py)
Demonstrates how to decompose a multivector into its even and odd grade parts. Even grades include scalars (grade 0) and bivectors (grade 2) in 3D VGA, while odd grades include vectors (grade 1) and trivectors (grade 3). This is fundamental for rotors and other geometric transformations.

#### How to run:
```bash
uv run python examples/algebra/even_odd_decomposition.py
```

#### Expected Output:
```text
=== Even / Odd Decomposition ===
Original multivector: [ 1.  -0.5  2.   3. ]
Even part: [2.]
Odd part: [ 1.  -0.5  3. ]

Recomposition:
[ 1.  -0.5  2.   3. ]
```
