# AMSA Benchmarks

Small benchmark scripts live here as lightweight future reference points.

Current scripts:

- `motor_ops.py` — PGA2d / PGA3d motor `exp` and `log` timing
- `backend_outputs.py` — dense, CSR, and JAX-preserving binary execution timing

Run from the repo root:

```bash
./.venv/bin/python benchmarks/motor_ops.py
./.venv/bin/python benchmarks/backend_outputs.py
```

Optional flags:

```bash
./.venv/bin/python benchmarks/motor_ops.py --number 5000 --repeat 7
./.venv/bin/python benchmarks/backend_outputs.py --number 1000 --repeat 7
```
