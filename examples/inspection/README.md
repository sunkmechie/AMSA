# Inspection Examples

Examples showing AMSA's inspection tools for understanding algebra structure,
product plans, IR execution, and multivector display.

## Examples

### [Cayley Table](cayley_table.py)
Displays the Cayley (multiplication) table for an algebra, showing how basis
blades multiply under the geometric product.

```bash
uv run python examples/inspection/cayley_table.py
```

### [MVArray Display](mvarray_display.py)
Shows how AMSA multivectors are pretty-printed — blade names, coefficients,
grades, and batch shapes.

```bash
uv run python examples/inspection/mvarray_display.py
```

### [OpPlan Inspection](plan_inspection.py)
Inspects the internal product plan for a geometric or outer product, showing
which blade pairs contribute to which output blades.

```bash
uv run python examples/inspection/plan_inspection.py
```

### [ProductIR Inspection](ir_inspection.py)
Inspects the intermediate representation (IR) of an operation — the instruction
sequence that the backend executes.

```bash
uv run python examples/inspection/ir_inspection.py
```
