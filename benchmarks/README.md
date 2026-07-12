# AMSA Benchmarks

Small benchmark scripts live here as lightweight future reference points.

Current scripts:

- `motor_ops.py` — PGA2d / PGA3d motor `exp` and `log` timing
- `ir_routing.py` — IR routing overhead for common operations (products, involutions)
- `storage_backends.py` — Dense vs CSR storage performance comparison
- `csr_native.py` — CSR-native indexing, broadcast add/sub, and product timing against dense baselines and old densify fallbacks
- `jax_traceability.py` — NumPy vs dense JAX eager, `jit`, `vmap`, and `grad` comparison

Run from the repo root:

```bash
./.venv/bin/python benchmarks/motor_ops.py
./.venv/bin/python benchmarks/ir_routing.py
./.venv/bin/python benchmarks/storage_backends.py
./.venv/bin/python benchmarks/csr_native.py
./.venv/bin/python benchmarks/jax_traceability.py
```

Run the fusion comparison benchmark:

```bash
uv run python benchmarks/fusion_comparison.py
```

## Benchmark Results

### CSR Native Paths

The CSR-native benchmark isolates paths that should stay sparse after CSR work:

```bash
uv run python benchmarks/csr_native.py --batch-size 2048 --number 200 --repeat 7
```

Use it when changing CSR indexing, broadcast add/sub, or product execution. Compare
`(csr native)` rows against `old densify fallback` rows for the direct regression
question, and against `dense baseline` rows to understand Python CSR overhead.
Tests separately enforce whether the operation preserves CSR output.

**Latest local run** (`batch_size=2048`, `number=200`, `repeat=7`):

```
getitem batch slice (csr native)                           best=   36.991 us  median=   38.634 us  mean=   40.293 us
getitem batch slice (dense baseline)                       best=    6.168 us  median=    6.264 us  mean=    6.472 us
getitem batch slice (old densify fallback)                 best= 1780.045 us  median= 1811.199 us  mean= 1852.459 us
add broadcast csr+csr (csr native)                         best= 2054.191 us  median= 2061.761 us  mean= 2065.815 us
add broadcast dense+dense baseline                         best=  101.862 us  median=  106.465 us  mean=  112.546 us
add broadcast old densify fallback                         best= 1947.406 us  median= 1953.173 us  mean= 1959.644 us
sub broadcast csr-csr (csr native)                         best= 2068.003 us  median= 2080.776 us  mean= 2084.456 us
sub broadcast dense-dense baseline                         best=  102.185 us  median=  102.505 us  mean=  102.986 us
sub broadcast old densify fallback                         best= 1953.170 us  median= 1969.485 us  mean= 1971.640 us
geometric_product csr*csr (csr native)                     best= 3583.194 us  median= 3594.228 us  mean= 3616.022 us
geometric_product dense*dense baseline                     best=  204.683 us  median=  205.528 us  mean=  205.619 us
geometric_product old densify fallback                     best= 3891.324 us  median= 3912.322 us  mean= 3922.750 us
```

Pass 1 removed internal CSR revalidation from helper-produced CSR arrays and
uses vectorized sparse-entry reduction for CSR/CSR add-sub. The run now shows
large wins for CSR slicing, a small win for CSR/CSR product versus old dense
fallback, and add-sub roughly tied with old fallback. Dense remains much faster
for this small-layout benchmark.


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
