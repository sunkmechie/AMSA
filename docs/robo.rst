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
- ``fk(alg, dh_params, *, joint_types)`` for CGA forward kinematics using
  Denavit–Hartenberg motor composition.

Forward kinematics
------------------

``fk()`` computes world-frame link positions for an N-DOF serial chain using
the CGA motor formulation from Bayro-Corrochano & Zamora-Esquivel (2007).
Each link-joint pair is defined by four DH parameters ``(α, a, d, θ)``:

.. code-block:: text

   M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

.. code-block:: python

   import amsa.robo as robo
   from amsa import Algebra

   alg = Algebra.cga3d()

   # Revolute joints: θ varies; prismatic: d varies
   results = robo.fk(alg, [
       (α₁, a₁, d₁, θ₁),   # joint 1
       (α₂, a₂, d₂, θ₂),   # joint 2
       ...
   ])

   for motor, tip in results:
       print(alg.extract_point(tip))

Parameters:

- ``α`` — link twist about x-axis (radians)
- ``a`` — link length along x-axis
- ``d`` — joint offset along z-axis
- ``θ`` — joint rotation about z-axis (radians, variable for revolute)

For prismatic joints pass ``joint_types=["prismatic", ...]``.

Citation: Bayro-Corrochano and Zamora-Esquivel (2007), "Differential and
inverse kinematics of robot devices using conformal geometric algebra",
Robotica 25(1), pp. 43–61.  See also Dorst et al. (2007), *GACS*, §15.5.

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
