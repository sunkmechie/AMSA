# AMSA Benchmarks

Small benchmark scripts live here as lightweight future reference points.

Current scripts:

- `motor_ops.py` — PGA2d / PGA3d motor `exp` and `log` timing
- `ir_routing.py` — IR routing overhead for common operations (products, involutions)
- `storage_backends.py` — Dense vs CSR storage performance comparison
- `jax_traceability.py` — NumPy vs dense JAX eager, `jit`, `vmap`, and `grad` comparison

Run from the repo root:

```bash
./.venv/bin/python benchmarks/motor_ops.py
./.venv/bin/python benchmarks/ir_routing.py
./.venv/bin/python benchmarks/storage_backends.py
./.venv/bin/python benchmarks/jax_traceability.py
```

Run the fusion comparison benchmark:

```bash
uv run python benchmarks/fusion_comparison.py
```

## Benchmark Results

### Dense JAX Traceability

The JAX traceability benchmark compares NumPy eager execution with dense JAX
eager and warmed compiled paths:

```bash
uv run python benchmarks/jax_traceability.py --batch-size 10000 --number 1000 --repeat 5
```

It reports:

- NumPy eager batch products and norm-squared baselines
- JAX eager batch product overhead
- warmed `jax.jit` batch products and norm-squared
- warmed `jax.jit(jax.vmap(...))` product execution
- warmed `jax.jit(jax.grad(...))` scalar-objective differentiation

The warmed JIT rows are the ones to use when checking whether the JAX
implementation is doing useful compiled coefficient work.

### Fusion Comparison (NumPy Backend)

The fusion comparison benchmark measures the performance impact of IR fusion optimizations in the NumPy backend.

**Results (100 iterations, 3 samples):**

```
Scale + Product (small):
scale+product (non-fused)                best=   48.280 us  median=   48.517 us  mean=   49.953 us
scale+product (fused)                    best=   38.246 us  median=   38.684 us  mean=   38.990 us

Scale + Product (large batch):
scale+product large batch (non-fused)    best=  108.165 us  median=  109.679 us  mean=  122.922 us
scale+product large batch (fused)        best=   94.925 us  median=   97.260 us  mean=   97.831 us
```

**Interpretation:**

- **Scale + Product fusion**: 12-21% faster by avoiding intermediate array allocation
- Fusion is kept focused on Clifford product sequences that avoid significant intermediate allocations

### Motor Operations

The motor operations benchmark measures the performance of motor exponential and logarithm operations for PGA2d and PGA3d.
