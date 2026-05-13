Autodiff Roadmap
================

``amsa.autodiff`` currently provides a deliberately naive forward-mode dual
surface:

- ``DualMV(real, tangent)`` pairs a primal multivector with one tangent
  direction.
- ``directional_derivative(fn, point, seed)`` evaluates a seeded derivative.
- ``forward_grad(fn, point)`` seeds one coefficient at a time for scalar
  objectives.

This reference layer is intentionally algebra-generic.  Product, projection,
involution, duality, contraction, and scalar-objective rules should apply across
all AMSA algebras because they are defined in terms of blades, layouts, and
public products.

Planned IR path
---------------

The optimized path should move in phases:

1. Keep ``DualMV`` as a correctness oracle for small tests and examples.
2. Add derivative metadata beside public op/IR construction, not inside
   robotics or other domain adapters.
3. Represent a forward-mode value as a primal/tangent pair of normal AMSA
   arrays, preserving algebra and layout metadata for both parts.
4. Route bilinear operations through product plans once for the primal and two
   tangent terms:

   ``d(A * B) = dA * B + A * dB``

5. Add specific rules for nonlinear operations such as normalization,
   inverse, ``exp()``, ``motor_exp()``, and ``motor_log()`` only when their
   algebraic domains are explicit.
6. Let backend implementations execute those rules for NumPy, JAX, and future
   backends without making domain packages depend on backend internals.

Robotics Boundary
-----------------

``amsa.robo`` should not depend on IR internals.  It should consume public AMSA
constructors, operations, and eventually public autodiff helpers.  This keeps a
future standalone ``amsa-robo`` package clean: robot loaders and solvers can
depend on AMSA as a Clifford engine, while AMSA's core IR remains independent of
robot file formats such as URDF, SRDF, and MJCF.
