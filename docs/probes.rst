Visual debugger probe
=====================

The ``probes/amsa_lab.py`` script is an external visual debugger for AMSA binary products. It traces operator plans and renders a self-contained HTML report. It intentionally lives outside ``src/amsa`` so that visualization and debugging UX can evolve without changing core algebra semantics.

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
