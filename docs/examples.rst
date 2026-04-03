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

Kernels
-------

- ``examples/kernels/geometric_kernels.py`` — inner-product kernel patterns

Planes
------

- ``examples/planes/point_plane_distance_3d.py`` — PGA plane/distance semantics

Robotics
--------

- ``examples/robotics/pga_circle_motion_2d.py``
- ``examples/robotics/pga_corridor_corner_2d.py``
- ``examples/robotics/pga_rigid_body_trajectory_2d.py``
- ``examples/robotics/planar_heading_rotor.py``
- ``examples/robotics/ray_plane_reflection_3d.py``
- ``examples/robotics/robot_rotation_comparison_2d.py``
- ``examples/robotics/trilateration_localization_2d.py``
- ``examples/robotics/vector_projection_2d.py``

All examples can be run directly:

.. code-block:: bash

   uv run python examples/geometry/triangle_area_2d.py
