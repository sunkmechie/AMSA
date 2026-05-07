References
==========

AMSA keeps detailed citations here. Docstrings should describe behavior and link back to this page
when an implementation depends on a published convention or algorithm.

Algebra
-------

- Dorst, Fontijne, and Mann, *Geometric Algebra for Computer Science*,
  Morgan Kaufmann, 2007.  General Clifford algebra conventions, versors,
  duality, and conformal models.
- Perwass, *Geometric Algebra with Applications in Engineering*, Springer,
  2009.  Engineering-oriented conventions for products, duality, and
  conformal geometry.

Operations
----------

- Dorst, Fontijne, and Mann, *Geometric Algebra for Computer Science*,
  §5.6.  De Morgan-style meet definitions via duality.
- Perwass, *Geometric Algebra with Applications in Engineering*, §4.3.4.
  Meet/regressive product via the dual of the join of duals.

Layouts
-------

- AMSA layouts are implementation descriptors: they specify blade ordering 
  and support. The blade semantics come from the algebra references above.

Storage
-------

- AMSA storage is coefficient representation only.  Dense and CSR storage
  preserve the same public multivector semantics.

CGA
---

- Dorst, Fontijne, and Mann, *Geometric Algebra for Computer Science*,
  Chapter 13, Tables 13.1-13.4.  Conformal point, sphere, plane, and
  Euclidean extraction conventions.
- Dorst, Fontijne, and Mann, *Geometric Algebra for Computer Science*,
  Chapter 15, §15.5.  Versors for Euclidean motion.
- Perwass, *Geometric Algebra with Applications in Engineering*, §4.3.2.
  Inverse mapping from conformal points to Euclidean coordinates.
- Hestenes, Li, and Rockwood, "New Algebraic Tools for Classical Geometry",
  in *Geometric Computing with Clifford Algebra*, Springer, 2001.  CGA
  incidence and geometric construction background.

Robotics
--------

- Bayro-Corrochano and Zamora-Esquivel, "Differential and inverse kinematics
  of robot devices using conformal geometric algebra", *Robotica* 25(1),
  2007, https://doi.org/10.1017/S0263574706002980.  CGA motor-based
  Denavit-Hartenberg serial-chain kinematics.
- Buss, "Introduction to Inverse Kinematics with Jacobian Transpose,
  Pseudoinverse and Damped Least Squares methods", 2004.  Numerical DLS IK.
- Wampler, "Manipulator Inverse Kinematic Solutions Based on Vector
  Formulations and Damped Least-Squares Methods", *IEEE Transactions on
  Systems, Man, and Cybernetics* 16(1), 1986.
- Nakamura and Hanafusa, "Inverse Kinematic Solutions With Singularity
  Robustness for Robot Manipulator Control", *Journal of Dynamic Systems,
  Measurement, and Control* 108(3), 1986.
- Siciliano, Sciavicco, Villani, and Oriolo, *Robotics: Modelling, Planning
  and Control*, Springer, 2010, §3.  Geometric Jacobian formulation.
- Shepperd, "Quaternion from Rotation Matrix", *Journal of Guidance and
  Control* 1(3), 1978.  Numerically stable matrix-to-quaternion conversion.
- Kleppe and Egeland, "Inverse Kinematics for Industrial Robots using
  Conformal Geometric Algebra", *Modeling, Identification and Control* 37(1),
  2016, https://doi.org/10.4173/mic.2016.1.6.
- Zaplana et al., "Closed-form solutions for the inverse kinematics of serial
  robots using conformal geometric algebra", *Mechanism and Machine Theory*
  173, 2022, https://doi.org/10.1016/j.mechmachtheory.2022.104835.
- Carbajal-Espinosa, Campos-Macias, and Diaz-Rodriguez, "FIKA: A Conformal
  Geometric Algebra Approach to a Fast Inverse Kinematics Algorithm for an
  Anthropomorphic Robotic Arm", *Machines* 12(1), 2024,
  https://doi.org/10.3390/machines12010078.

Backends And JAX
----------------

- JAX documentation:

  - ``jax.jit``: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
  - Custom pytrees: https://docs.jax.dev/en/latest/custom_pytrees.html
  - ``jax.vmap``: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
  - JAX errors: https://docs.jax.dev/en/latest/errors.html

Visualization
-------------

- AMSA visualization is adapter based.  It exposes geometric primitives and
  leaves backend choice to users, for example Matplotlib for plots and VisPy
  for interactive scenes.
