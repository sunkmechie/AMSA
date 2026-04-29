# JAX Examples

This directory contains examples showing how dense AMSA multivectors work with JAX transformations while keeping blades, layouts, and product plans as Clifford metadata.

## Examples

### [Dense Traceability](dense_traceability.py)
Demonstrates:
- `jax.jit` over a dense geometric product
- `jax.vmap` over dense multivector coefficient leaves
- `jax.grad` through a scalar norm-squared objective

The example uses AMSA multivectors directly as JAX pytrees. Coefficient arrays are dynamic JAX values; algebra and layout metadata stay static.

#### How to run:
```bash
uv run python examples/jax/dense_traceability.py
```

#### Expected Output:
```text
=== Dense JAX Traceability ===
jitted geometric product:
...

jitted vmap outer product:
...

norm-squared gradient:
...
```
