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

Optional flags:

```bash
./.venv/bin/python benchmarks/motor_ops.py --number 5000 --repeat 7
./.venv/bin/python benchmarks/ir_routing.py --number 5000 --repeat 7
./.venv/bin/python benchmarks/storage_backends.py --number 2000 --repeat 7
./.venv/bin/python benchmarks/backend_comparison.py --number 1000 --repeat 5
```

**Note:** The backend comparison benchmark requires JAX to be installed. For CPU execution: `uv pip install amsa-ga[jax]`. For GPU execution (CUDA): `uv pip install "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`.

**Benchmark interpretation:**
- This benchmark measures **latency** (per-call overhead) through the AMSA abstraction layer
- NumPy is expected to win for small/single operations (minimal overhead)
- JAX may appear slower due to tracing overhead and Python object wrapping
- True JAX performance (JIT compilation, GPU throughput) requires deeper backend integration (future work)
- Large batch tests show throughput potential, but still go through Python abstraction
