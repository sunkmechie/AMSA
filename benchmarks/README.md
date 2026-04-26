# AMSA Benchmarks

Small benchmark scripts live here as lightweight future reference points.

Current scripts:

- `motor_ops.py` — PGA2d / PGA3d motor `exp` and `log` timing
- `ir_routing.py` — IR routing overhead for common operations (products, involutions)
- `storage_backends.py` — Dense vs CSR storage performance comparison
- `backend_comparison.py` — NumPy vs JAX backend performance comparison

Run from the repo root:

```bash
./.venv/bin/python benchmarks/motor_ops.py
./.venv/bin/python benchmarks/ir_routing.py
./.venv/bin/python benchmarks/storage_backends.py
./.venv/bin/python benchmarks/backend_comparison.py
```

Run the fusion comparison benchmark:

```bash
uv run python benchmarks/fusion_comparison.py
```

## Benchmark Results

### Backend Comparison (NumPy vs JAX)

The backend comparison benchmark measures the performance difference between NumPy and JAX backends for common operations.

**Important Notes:**

- This benchmark measures latency through the AMSA abstraction layer, not raw backend performance
- True JAX performance requires deeper backend integration (JIT compilation through Clifford composition)
- Current JAX backend uses jax.numpy but cannot trace through AMSA objects
- For meaningful JAX performance, use large batched operations and enable JIT where applicable
- The benchmark is useful for comparing relative performance, not absolute throughput

### Fusion Comparison (NumPy Backend)

The fusion comparison benchmark measures the performance impact of IR fusion optimizations in the NumPy backend.

**Results (100 iterations, 3 samples):**

```
Scale + Product (small):
scale+product (non-fused)                best=   48.280 us  median=   48.517 us  mean=   49.953 us
scale+product (fused)                    best=   38.246 us  median=   38.684 us  mean=   38.990 us

Elementwise Chain (small):
elementwise chain (non-fused)            best=    0.685 us  median=    0.688 us  mean=    0.753 us
elementwise chain (fused)                best=    1.006 us  median=    1.020 us  mean=    1.038 us

Scale + Product (large batch):
scale+product large batch (non-fused)    best=  108.165 us  median=  109.679 us  mean=  122.922 us
scale+product large batch (fused)        best=   94.925 us  median=   97.260 us  mean=   97.831 us
```

**Interpretation:**

- **Scale + Product fusion**: 12-21% faster by avoiding intermediate array allocation
- **Elementwise Chain fusion**: 47% slower for simple operations due to function call overhead
- Fusion is beneficial for operations that avoid significant intermediate allocations
- Elementwise fusion may become beneficial for longer chains or larger arrays

### Motor Operations

The motor operations benchmark measures the performance of motor exponential and logarithm operations for PGA2d and PGA3d.
