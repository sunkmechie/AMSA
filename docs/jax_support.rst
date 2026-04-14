# JAX Backend Support Matrix

## Current Status: Beta (Partial)

AMSA's JAX backend is production-ready for **binary operators and JIT compilation**, but lacks support for automatic differentiation and advanced JAX transforms. This document clarifies what works, what doesn't, and the path to full support.

## What Works ✅

### Storage Backend
- JAX array construction and storage (immutable, GPU-compatible)
- Conversion between dense, CSR, and JAX backends
- Batch dimension support
- Storage-local operations (scale, project, reweight)

### Operators (JAX-Compatible)
- **Binary Products** (JIT-compiled): geometric, outer, inner, scalar, left/right contraction, regressive, commutator, anticommutator
- **Unary Operations** (Python execution): reverse, involute, conjugate, dual, grade projection, norms, normalization
- **Higher-Level Operations** (Python execution): inverse, sandwich, exp, motor_exp, add, sub, neg
- **PGA-Specific** (Python execution): bulk, weight, bulk/weight norms, bulk/weight normalization

### Tracing Support
- PyTree registration enables `@jax.jit` fusion of composite expressions
- Eager-mode validation ensures algebraic semantics are preserved (no silent semantic changes)
- Tracer-safe implementations in exp(), inverse() and related scalar-requiring operations

### Correctness
- JAX eager mode preserves exact AMSA algebraic semantics (same errors as dense)
- Batched operations with broadcasting
- Numerical accuracy on par with NumPy dense backend

## What Doesn't Work ❌

### Automatic Differentiation
- No `grad()`, `jvp()`, `vjp()` support
- Cannot use in optimization loops or neural networks requiring gradients
- Blocks ML/robotics control workflows

### Advanced JAX Transforms
- No `vmap()` support (manual batching only)
- No `scan()` or `while_loop()` (no sequential operation unrolling)
- No `cond()` (no branching in XLA)
- No custom XLA kernels
- No `pmap()` (no multi-device parallelism)

### Control Flow
- Complex algorithms with dynamic control flow cannot be compiled
- No adaptive precision or numerical error handling within compiled regions

## System Requirements

### Python & JAX Version
- Python 3.10+
- JAX 0.4.0+ (tested with 0.4.x)
- NumPy 1.20+

### JAX Configuration
**Critical**: AMSA requires float64 support. Set before JAX import:
```bash
export JAX_ENABLE_X64=1
```

Or in Python:
```python
import os
os.environ["JAX_ENABLE_X64"] = "1"

# Then import AMSA
from amsa import Algebra
```

Without x64 enabled:
- float64 arrays silently truncate to float32
- Numerical accuracy degrades
- JAXStorage.zeros() raises ValueError

### Device Support
- CPU: Fully supported
- GPU: Fully supported (CUDA/ROCm/Metal backends)
- TPU: Supported via standard JAX setup

## Performance Characteristics

### Latency (Single Operations)
- **Overhead**: JAX kernel launch ~100μs on GPU, ~1μs on CPU
- **Recommendation**: For single small operations, NumPy dense is faster

### Throughput (Batched Operations)
- **Sweet Spot**: 1,000+ elements per batch
- **Scaling**: 10,000x speedup possible on large batches vs NumPy
- **GPU**: Enables parallelism; highly recommended for batch size > 1,000

### Compilation Time
- First call: 10–100ms (JIT compilation)
- Subsequent calls: Cached kernel reuse
- **Tip**: Compile hot paths once, reuse many times

## Behavioral Differences

### Eager Mode (No @jax.jit)
- **Semantics**: Identical to NumPy dense
- **Execution**: Python control flow executes; JAX arrays flow through
- **Validation**: All checks (scalar requirements, invertibility) enforced immediately

### Traced Mode (@jax.jit)
- **Semantics**: Algebraic semantics preserved, but tracing context allows deferred validation
- **Execution**: Python control flow recorded into trace; only deterministic ops execute in XLA
- **Validation**: Scalar and invertibility checks deferred to compiled region (may produce inf/nan at runtime if violated)

## Known Limitations

1. **Scalar-Only Operations Are Lossy in Tracing**: Operations like `inverse()` and `exp()` skip validation during tracing to support deferred evaluation. Invalid inputs may produce `inf`/`nan` at runtime instead of raising errors.

2. **Mixed Backend Fallback**: JAX + NumPy operations fall back to NumPy (logged, not silent).

3. **No Autodiff Pipeline**: `grad()` on AMSA operations is not implemented; numerical gradients possible but not efficient.

4. **Limited Control Flow**: No `cond()` or `while_loop()` support; all loops must be fully unrolled at trace time.

## Production Readiness by Workload

| Workload | Status | Notes |
|----------|--------|-------|
| Batch geometric computations | ✅ Stable | Binary operators JIT'ed, high throughput |
| Rigid body transformations | ✅ Stable | PGA batch operations tested |
| Visualization + GPU acceleration | ✅ Stable | PyTree fusion enables efficient rendering |
| Real-time single-op serving | ⚠️ Marginal | Kernel launch overhead; consider CPU NumPy |
| Neural network layers | ❌ Blocked | No autodiff; not feasible |
| Optimal trajectory planning | ❌ Blocked | No `scan()`; no `grad()` |
| Adaptive control | ❌ Blocked | No `cond()`; no dynamic branching |

## Path to Full Support (Estimated 3–4 Months)

1. **Autodiff** (4 weeks): Implement `grad()`, `jvp()`, `vjp()` via JAX primitives
2. **Vmap** (3 weeks): Register vectorization rules for all operators
3. **Control Flow** (5 weeks): Add `cond()`, `while_loop()`, `scan()` support
4. **Testing** (2 weeks): CI on GPU/TPU, numerical stability validation

This would unblock ML/robotics workflows requiring optimization and adaptive algorithms.

## Troubleshooting

### `JAXStorage requires float64 support...` Error
- **Cause**: JAX_ENABLE_X64 not set before import
- **Fix**: `export JAX_ENABLE_X64=1` before running Python

### `TracerArrayConversionError` in traced code
- **Cause**: Trying to materialize a tracer value (e.g., `.item()` in a traced function)
- **Fix**: Use JAX numpy functions (`jnp.*`) instead of NumPy inside `@jax.jit`

### Float32 results instead of float64
- **Cause**: JAX_ENABLE_X64 disabled; values silently truncate
- **Fix**: Enable x64 mode; check `.dtype` of JAX arrays

### Traced exp() / inverse() producing inf/nan
- **Cause**: Invalid input (non-scalar, non-invertible) in traced context; validation was deferred
- **Fix**: Validate inputs eagerly before tracing, or add explicit checks in trace

## See Also
- [storage.rst](storage.rst) — Backend selection and performance tuning
- [examples/](../examples/) — Example workloads
