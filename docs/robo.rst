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

   result = robo.ik_dls(alg, dh_params, target_motor,
                         joint_limits=limits,
                         position_tolerance=1e-6,
                         orientation_tolerance=1e-6)

Current scope:

- ``Link``, ``Joint``, and ``RobotModel`` dataclasses.
- ``importurdf(path)`` for URDF topology import.
- ``dump_crobot(model)`` and ``load_crobot(path)`` for the draft Clifford-native
  robot JSON shape.
- ``ik(..., solver="planar_two_link")`` as a minimal smoke-test solver.
- ``ik(..., solver="cga_sphere_sphere")`` for a CGA sphere-sphere reach meet.
- ``ik(..., solver="cga_point_circle")`` for selecting a point on a direct CGA
  circle.
- ``fk(alg, dh_params, *, joint_types)`` for CGA forward kinematics using
  Denavit–Hartenberg motor composition.
- ``ik_dls(alg, dh_params, target_motor, ...)`` for damped least-squares
  numerical inverse kinematics with adaptive damping and joint limits.
- ``IKResult`` dataclass with position, orientation, and convergence metadata.
- ``motor_to_position(motor, alg)``, ``motor_to_quaternion(motor, alg)``,
  ``motor_to_matrix(motor, alg)`` for extracting Cartesian pose from CGA motors.

Forward kinematics
------------------

``fk()`` computes world-frame Cartesian poses for an N-DOF serial DH chain using
CGA motor composition.  References are collected in :doc:`references`.
Each link-joint pair is defined by four DH parameters ``(α, a, d, θ)``:

.. code-block:: text

   M_i = M_{i-1} · T_z(d) · R_z(θ) · T_x(a) · R_x(α)

Returns a list of dictionaries, one per link:

.. code-block:: python

   import amsa.robo as robo
   from amsa import Algebra

   alg = Algebra.cga3d()

   results = robo.fk(alg, [
       (α₁, a₁, d₁, θ₁),   # joint 1
       (α₂, a₂, d₂, θ₂),   # joint 2
       ...
   ])

   ee = results[-1]
   pos = ee["position"]       # np.ndarray (x, y, z)
   quat = ee["orientation"]   # np.ndarray (w, x, y, z)
   motor = ee["motor"]        # MVArray (full CGA motor)

Parameters
^^^^^^^^^^

- ``α`` — link twist about x-axis (radians)
- ``a`` — link length along x-axis (metres)
- ``d`` — joint offset along z-axis (metres)
- ``θ`` — joint rotation about z-axis (radians, variable for revolute)

For prismatic joints pass ``joint_types=["prismatic", ...]``.

Motor-to-pose helpers
^^^^^^^^^^^^^^^^^^^^^

The full CGA motor is returned in every result dictionary.  Use these helpers
to extract pose components:

.. code-block:: python

   from amsa.robo import motor_to_position, motor_to_quaternion, motor_to_matrix

   pos = motor_to_position(motor, alg)     # (x, y, z) ndarray
   quat = motor_to_quaternion(motor, alg)  # (w, x, y, z) quaternion
   R = motor_to_matrix(motor, alg)         # 3×3 rotation matrix

- ``motor_to_position`` — applies the motor to the conformal origin.
- ``motor_to_quaternion`` — recovers the rotation matrix from basis vectors,
  then converts to a unit quaternion.
- ``motor_to_matrix`` — sandwiches the motor with the canonical Euclidean
  basis vectors ``e₁, e₂, e₃`` to recover the rotation matrix columns.

Inverse kinematics
------------------

``ik_dls()`` solves the inverse kinematics problem for a DH-parameterised
serial chain using the damped least-squares (Levenberg-Marquardt) method.
It takes a target end-effector CGA motor and returns joint angles that
drive the forward kinematics to that pose.

.. code-block:: python

   import amsa.robo as robo
   from amsa import Algebra

   alg = Algebra.cga3d()

   # UR5 DH parameters
   dh_params = [
       (π/2,  0.0,     0.089159, 0.0),
       (0.0, -0.42500, 0.0,      0.0),
       (0.0, -0.39225, 0.0,      0.0),
       (π/2,  0.0,     0.10915,  0.0),
       (-π/2, 0.0,     0.09465,  0.0),
       (0.0,  0.0,     0.08230,  0.0),
   ]

   # Compute a target motor via FK
   target = robo.fk(alg, dh_target)[-1]["motor"]

   # Solve IK
   result = robo.ik_dls(
       alg,
       dh_params,
       target,
       joint_limits=[(-2π, 2π)] * 6,
       position_tolerance=1e-6,
       orientation_tolerance=1e-6,
   )

   if result.success:
       print("Converged in", result.iterations, "iterations")
       print("Joint angles:", result.joint_angles)
       print("Position:", result.position)
       print("Orientation (quaternion):", result.orientation)

Algorithm
^^^^^^^^^

At each iteration the solver:

1. Runs FK to obtain per-joint world-frame positions, Z-axis directions,
   and the current end-effector pose.
2. Computes the 6-vector task-space error:
   ``e = [p_target - p_current,  axis_angle(R_target @ R_current^T)]``.
3. Builds the 6×n geometric Jacobian *J*:

   - Revolute column: ``[z_i × (p_ee − p_i);  z_i]``
   - Prismatic column: ``[z_i; 0]``

4. Applies the Levenberg-Marquardt update:

   .. math::

      Δθ = J^\\mathsf{T} (J J^\\mathsf{T} + λ^2 I)^{-1} e

5. Steps forward and adjusts the damping factor λ:

   - Error decreases → multiply λ by ``damping_factor`` (reduce damping).
   - Error increases → multiply λ by 2 (increase damping, reject step).

6. Clamps joint angles to ``joint_limits``.

Parameters
^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``alg``
     - ``Algebra``
     - required
     - CGA algebra instance (e.g. ``Algebra.cga3d()``).
   * - ``dh_params``
     - ``list[tuple]``
     - required
     - Denavit-Hartenberg ``(α, a, d, θ)`` tuples.
   * - ``target_motor``
     - ``MVArray``
     - required
     - Desired end-effector pose as a CGA motor.
   * - ``joint_types``
     - ``list[str]``
     - ``["revolute"] * n``
     - ``"revolute"`` or ``"prismatic"`` per joint.
   * - ``initial_angles``
     - ``ndarray``
     - ``zeros(n)``
     - Starting joint configuration.
   * - ``max_iterations``
     - ``int``
     - ``100``
     - Maximum solver iterations.
   * - ``position_tolerance``
     - ``float``
     - ``1e-6``
     - Convergence threshold for position error (metres).
   * - ``orientation_tolerance``
     - ``float``
     - ``1e-6``
     - Convergence threshold for orientation error (radians).
   * - ``damping``
     - ``float``
     - ``0.1``
     - Initial Levenberg-Marquardt damping λ.
   * - ``damping_factor``
     - ``float``
     - ``0.5``
     - Multiplier for adaptive damping schedule.
   * - ``joint_limits``
     - ``list[tuple]``
     - ``None``
     - Per-joint (min, max) bounds in radians or metres.

IKResult
^^^^^^^^

.. code-block:: python

   @dataclass
   class IKResult:
       success: bool                # converged within tolerances?
       joint_angles: np.ndarray     # solved angles
       motor: MVArray | None        # CGA motor at solution
       position: np.ndarray | None  # (x, y, z) end-effector position
       orientation: np.ndarray | None  # (w, x, y, z) quaternion
       iterations: int              # iterations taken
       position_error: float        # final position error (metres)
       orientation_error: float     # final orientation error (radians)
       converged_position: bool     # position within tolerance?
       converged_orientation: bool  # orientation within tolerance?

References
^^^^^^^^^^

See :doc:`references` for the DLS, geometric Jacobian, and CGA motor-DH
references.

Solver comparison
^^^^^^^^^^^^^^^^^

AMSA currently has two kinds of IK support:

- ``ik_dls()`` is the full serial-chain solver. It uses a damped least-squares
  update over the geometric Jacobian and targets a complete CGA end-effector
  motor. See :doc:`references` for the DLS, singularity-robust IK, geometric
  Jacobian, and CGA motor-DH references.
- ``ik(..., solver="cga_sphere_sphere")`` and
  ``ik(..., solver="cga_point_circle")`` are geometric IK primitives. They are
  useful for reach and branch-selection subproblems, but they are not yet a
  complete closed-form UR5 solver.

The runnable example
``examples/robotics/cga_ik_ur5_solver_comparison.py`` compares these surfaces
on UR5 geometry: DLS solves the full end-effector pose, while the CGA primitive
solvers recover elbow geometry from the same target chain.

Draft ``.crobot`` direction
---------------------------

URDF is still useful as an interchange bridge, but AMSA's native robotics format
will describe robot geometry in Clifford terms:

- Algebra model: ``cga3d`` for points, spheres, lines, planes, motors, and joint
  constraints.
- Links: named rigid bodies with optional conformal attachment points and
  collision primitives.
- Joints: parent/child relation plus a Clifford motion generator such as a
  rotation-axis line bivector or translation direction.
- Limits: scalar ranges attached to generator parameters.
- Frames: derived from motors, not stored as homogeneous matrices.

This draft exists to keep the future ``amsa-robo`` surface aligned with AMSA's 
blade/layout/storage boundaries before the package is split out.
