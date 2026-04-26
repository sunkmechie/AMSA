# AMSA Benchmarks

Small benchmark scripts live here as lightweight future reference points.

Current scripts:

- `motor_ops.py` — PGA2d / PGA3d motor `exp` and `log` timing
- `ir_routing.py` — IR routing overhead for common operations (products, involutions)
- `storage_backends.py` — Dense vs CSR storage performance comparison

Run from the repo root:

```bash
./.venv/bin/python benchmarks/motor_ops.py
./.venv/bin/python benchmarks/ir_routing.py
./.venv/bin/python benchmarks/storage_backends.py
```

Optional flags:

```bash
./.venv/bin/python benchmarks/motor_ops.py --number 5000 --repeat 7
./.venv/bin/python benchmarks/ir_routing.py --number 5000 --repeat 7
./.venv/bin/python benchmarks/storage_backends.py --number 2000 --repeat 7
```
