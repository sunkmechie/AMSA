Experimental Robotics
=====================

``amsa.robo`` is experimental and not ready for production robotics use. The
namespace is a staging area for a future standalone ``amsa-robo`` package while
the Clifford-native model, file format, and solver APIs settle.

Current API
-----------

.. code-block:: python

   import amsa.robo as robo

   model = robo.importurdf("robot.urdf")
   data = robo.dump_crobot(model)
   q1, q2 = robo.ik((1.0, 1.0), (1.0, 1.0), solver="planar_two_link")

Current scope:

- ``Link``, ``Joint``, and ``RobotModel`` dataclasses.
- ``importurdf(path)`` for URDF topology import.
- ``dump_crobot(model)`` and ``load_crobot(path)`` for the draft Clifford-native
  robot JSON shape.
- ``ik(..., solver="planar_two_link")`` as a minimal smoke-test solver.

Draft ``.crobot`` direction
---------------------------

URDF is still useful as an interchange bridge, but AMSA's native robotics format
should describe robot geometry in Clifford terms:

- Algebra model: ``cga3d`` for points, spheres, lines, planes, motors, and joint
  constraints.
- Links: named rigid bodies with optional conformal attachment points and
  collision primitives.
- Joints: parent/child relation plus a Clifford motion generator such as a
  rotation-axis line bivector or translation direction.
- Limits: scalar ranges attached to generator parameters.
- Frames: derived from motors, not stored as homogeneous matrices.

This draft deliberately avoids promising complete execution support. It exists
to keep the future ``amsa-robo`` surface aligned with AMSA's blade/layout/storage
boundaries before the package is split out.
