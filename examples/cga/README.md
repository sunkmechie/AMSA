# CGA Examples

Examples demonstrating Conformal Geometric Algebra (CGA) with AMSA.

CGA embeds Euclidean geometry into a higher-dimensional conformal space. It
treats points, spheres, planes, lines, and circles uniformly as blades — no
separate point/vector/line representations needed.

## Examples

### [CGA Primitives](cga_primitives.py)
Constructs all CGA geometry primitives in both 2D and 3D: null basis vectors
(n_o, n_inf), Euclidean vectors, conformal points, dual spheres, dual planes,
direct lines, direct circles, and translators. Prints blade decompositions and
verifies key identities (nullity, squared norms).

```bash
uv run python examples/cga/cga_primitives.py
```

### [Batched Point Distance](cga_point_distance_batch.py)
Constructs a batch of random 3D points using ``alg.point()`` and computes the
full pairwise distance matrix via ``alg.distance_squared()``. Verifies against
direct Euclidean computation.

```bash
uv run python examples/cga/cga_point_distance_batch.py
```

### [Classification Overview](cga_classify_overview.py)
Runs every CGA primitive (cga3d and cga2d) through ``alg.classify()`` and prints
the ``EntityInfo`` output — kind, grades, nullity, normalization, invariants,
geometric data, and storage metadata.

```bash
uv run python examples/cga/cga_classify_overview.py
```

### [Versor Actions](cga_versor_actions.py)
Applies translation and reflection versor actions, classifying the result each
time.  Shows that ``alg.classify()`` correctly identifies transformed points and
extracts the updated coordinates.

```bash
uv run python examples/cga/cga_versor_actions.py
```
