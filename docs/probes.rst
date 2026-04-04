Visual debugger probe
=====================

The ``probes/amsa_lab.py`` script is an external visual debugger for AMSA binary products. It traces operator plans and renders a self-contained HTML report. It intentionally remains separate from the lightweight in-package ``amsa.viz`` layer so that debugging UX can evolve without forcing the core visualization API to change with it.

What it does
------------

- Runs a trusted local AMSA expression
- Captures binary product steps from the existing operator path
- Uses :class:`amsa.plans.OpPlan` terms as the structural blade-interaction graph
- Renders an interactive HTML board with draggable nodes and live SVG wires

Limitations
-----------

- It is not a hardened expression parser
- It does not trace unary operations or addition/subtraction
- It does not include a geometry overlay for PGA object semantics

Relationship to ``amsa.viz``
----------------------------

AMSA now has two distinct visualization surfaces:

- ``amsa.viz`` for lightweight multivector-to-primitive adapters and simple plotting
- ``probes/amsa_lab.py`` for operator-plan introspection and debugger-style exploration

They serve different purposes and should not be treated as duplicates.

Usage
-----

.. code-block:: bash

   uv run python probes/amsa_lab.py \
     --algebra vga2d \
     --stmt "u = alg.vector([1.0, 2.0])" \
     --stmt "v = alg.vector([3.0, 4.0])" \
     --expr "u * v" \
     --output tempo/amsa_lab.html

Open ``tempo/amsa_lab.html`` in a browser to inspect the plan graph.
