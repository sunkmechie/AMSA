Examples
========

The ``examples/`` directory contains runnable scripts that demonstrate the public API across algebra, geometry, and robotics.

Algebra
-------

- ``examples/algebra/even_odd_decomposition.py`` — even/odd grade splitting

Geometry
--------

- ``examples/geometry/triangle_area_2d.py`` — 2D area via outer product
- ``examples/geometry/signed_volume_3d.py`` — scalar triple product
- ``examples/geometry/orientation_batch_2d.py`` — batched orientation tests

Inspection
----------

- ``examples/inspection/mvarray_display.py`` — MVArray human-readable display
- ``examples/inspection/plan_inspection.py`` — OpPlan structure inspection
- ``examples/inspection/ir_inspection.py`` — ProductIR execution inspection
- ``examples/inspection/cayley_table.py`` — Cayley table algebra inspection

JAX
---

- ``examples/jax/dense_traceability.py`` — dense ``jit``, ``vmap``, and ``grad`` usage

Kernels
-------

- ``examples/kernels/geometric_kernels.py`` — inner-product kernel patterns

Planes
------

- ``examples/planes/point_plane_distance_3d.py`` — PGA plane/distance semantics

Robotics
--------

- ``examples/robotics/pga_corridor_corner_2d.py`` — wall intersection via regressive product
- ``examples/robotics/pga_rigid_body_trajectory_2d.py`` — repeated motor application
- ``examples/robotics/pga3d_kinematic_chain.py`` — 3D PGA kinematic chain
- ``examples/robotics/planar_heading_rotor.py`` — VGA rotor heading update
- ``examples/robotics/ray_plane_reflection_3d.py`` — sandwich reflection in VGA3d
- ``examples/robotics/robot_rotation_comparison_2d.py`` — rotor vs matrix/quaternion comparison
- ``examples/robotics/trilateration_localization_2d.py`` — PGA2d localization with optional viz
- ``examples/robotics/vector_projection_2d.py`` — corridor-axis projection

All examples can be run directly:

.. code-block:: bash

   uv run python examples/geometry/triangle_area_2d.py

Some robotics examples use ``amsa.viz.backends.mpl`` for 2D plots and
``amsa.viz.backends.vispy`` for 3D interactive scenes.
