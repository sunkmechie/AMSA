Inspection API
==============

AMSA provides inspection and pretty-print helpers for debugging and understanding algebraic operations without symbolic computation overhead.

MVArray Display
---------------

The ``MVArray.__repr__`` method provides human-readable multivector representations:

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga2d()
   u = alg.vector([1.0, 2.0])
   v = alg.bivector([3.0])

   print(u)  # 1.0 e1 + 2.0 e2
   print(v)  # 3.0 e12

For batched multivectors, the repr shows batch metadata:

.. code-block:: python

   batch = alg.zeros(batch_shape=(2, 3))
   print(batch)  # MVArray(batch_shape=(2, 3), blades=0, dtype=float64)

Plan Inspection
---------------

Use ``OpPlan.show()`` to inspect product plans in algebra notation:

.. code-block:: python

   from amsa import Algebra
   from amsa.plans import plan_binary_product

   alg = Algebra.vga2d()
   lhs_layout = alg.grade_layout(1)
   rhs_layout = alg.grade_layout(1)
   plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")

   print(plan.show())
   # OpPlan(geometric)
   #   LHS blades: e1, e2
   #   RHS blades: e1, e2
   #   Output blades: e, e1, e2, e12
   #   Terms (4):
   #     + e1 * e1 -> e
   #     + e1 * e2 -> e12
   #     + e2 * e1 -> -e12
   #     + e2 * e2 -> e

IR Inspection
-------------

Use ``ProductIR.show()`` to inspect the storage-aware IR:

.. code-block:: python

   from amsa import Algebra
   from amsa.plans import plan_binary_product
   from amsa.ir import build_product_ir

   alg = Algebra.vga2d()
   lhs_layout = alg.grade_layout(1)
   rhs_layout = alg.grade_layout(1)
   plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")
   ir = build_product_ir(plan, "dense", "dense")

   print(ir.show(alg.spec))
   # ProductIR(geometric)
   #   LHS storage: dense, width: 2
   #   RHS storage: dense, width: 2
   #   Output blades: e, e1, e2, e12
   #   Terms (4):
   #     + col[0] * col[0] -> col[0]
   #     + col[0] * col[1] -> col[3]
   #     + col[1] * col[0] -> col[3]
   #     + col[1] * col[1] -> col[1]

Algebra Inspection
------------------

Use ``Algebra.show_cayley()`` to display Cayley table subsets:

.. code-block:: python

   from amsa import Algebra

   alg = Algebra.vga2d()
   print(alg.show_cayley())
   # Cayley table for (1, 1) (4 blades)
   #       e    e1   e2   e12
   #   e    e    e1   e2   e12
   #   e1   e1   e    e12  e2
   #   e2   e2  -e12  e   -e1
   #   e12  e12  -e2  e1   e

For custom blade selection:

.. code-block:: python

   print(alg.show_cayley(blades=(0, 1, 2)))
   # Cayley table for (1, 1) (3 blades)
   #       e    e1   e2
   #   e    e    e1   e2
   #   e1   e1   e    e12
   #   e2   e2  -e12  e
