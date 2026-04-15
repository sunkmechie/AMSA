"""Intermediate representation for AMSA execution backends.

This module defines storage-aware, backend-neutral IR descriptors that
capture the computation graph of Clifford algebra operations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from amsa.layouts import MVLayout
from amsa.plans import OpKind, OpPlan
from amsa.specs import AlgebraSpec, grade_of_blade
from amsa.storage import StorageKind

UnaryKind = Literal[
    "reverse",
    "involute",
    "conjugate",
    "dual",
    "undual",
    "poincare_dual",
    "poincare_undual",
]



SequenceStepKind = Literal[
    "binary_product",
    "unary",
    "scale",
    "row_scale",
    "add",
    "sub",
    "neg",
    "scalar_extract",
    "single_blade_mv",
]




@dataclass(frozen=True, slots=True)
class TermIR:
    """A single coefficient multiply-accumulate term.

    Attributes:
        lhs_col: Layout-local column index in the LHS storage.
        rhs_col: Layout-local column index in the RHS storage.
        out_col: Layout-local column index in the output storage.
        coefficient: ±1 or a metric-derived integer (never zero).
    """

    lhs_col: int
    rhs_col: int
    out_col: int
    coefficient: int


@dataclass(frozen=True, slots=True)
class ProductIR:
    """Lowered IR for a binary multivector product.

    This is the storage-aware counterpart of ``plans.OpPlan``.  Where
    ``OpPlan`` uses blade bit-patterns (algebra-level identifiers),
    ``ProductIR`` uses layout-local column indices so that backends can
    operate directly on storage without consulting layout metadata.

    Attributes:
        kind: The operator kind (geometric, outer, inner, …).
        lhs_storage: Storage kind of the LHS operand.
        rhs_storage: Storage kind of the RHS operand.
        lhs_width: Number of columns in the LHS layout.
        rhs_width: Number of columns in the RHS layout.
        out_blades: Blade bit-patterns of the output layout, in order.
        terms: Flatten multiply-accumulate terms.
    """

    kind: OpKind
    lhs_storage: StorageKind
    rhs_storage: StorageKind
    lhs_width: int
    rhs_width: int
    out_blades: tuple[int, ...]
    terms: tuple[TermIR, ...]

    @property
    def out_width(self) -> int:
        """Number of columns in the output layout."""
        return len(self.out_blades)



@dataclass(frozen=True, slots=True)
class UnaryIR:
    """Lowered IR for a unary multivector operation.

    Two mutually exclusive encodings are used:

    - **Weights**: per-column scalar multipliers.  This covers ``reverse``,
      ``involute``, ``conjugate``, and the pseudoscalar-weighted dual/undual
      variants.
    - **Permutation**: a source→column map where ``permutation[out_col]``
      gives the source column to read from.  Combined with ``weights`` for
      the metric sign on each permuted entry.  This covers ``dual``,
      ``undual``, ``poincare_dual``, and ``poincare_undual``.

    Attributes:
        kind: The unary operation kind.
        input_width: Number of columns in the input layout.
        out_blades: Blade bit-patterns of the output layout, in order.
        weights: Per-column weights (length == ``out_width``).
        permutation: Source-column map (length == ``out_width``).
            ``None`` when the operation is a pure per-column weight
            (reverse, involute, conjugate).
    """

    kind: UnaryKind
    input_width: int
    out_blades: tuple[int, ...]
    weights: tuple[float, ...]
    permutation: tuple[int, ...] | None = None

    @property
    def out_width(self) -> int:
        """Number of columns in the output layout."""
        return len(self.out_blades)

    @property
    def is_permutation(self) -> bool:
        """True when this IR encodes a complement permutation (dual/undual)."""
        return self.permutation is not None



@dataclass(frozen=True, slots=True)
class IRStep:
    """A single step inside a ``SequenceIR`` computation graph.

    Attributes:
        kind: The step operation kind.
        operands: Named references consumed by this step.
            Common names: ``"lhs"``, ``"rhs"``, ``"input"``, ``"temp_N"``,
            ``"scalar"``.  For ``"binary_product"`` the convention is
            ``("lhs", "rhs")``.  For ``"unary"`` it is ``("input",)``.
        ir: The sub-IR that describes this step's internal computation.
            - ``ProductIR`` for ``"binary_product"``
            - ``UnaryIR`` for ``"unary"``
            - ``None`` for primitive steps (``"scale"``, ``"add"``, …)
        output: Named reference where the result is stored for downstream
            steps to consume.
        metadata: Optional opaque dict for backend-specific hints
            (e.g., ``{"function": "sqrt"}`` for elementwise ops).
    """

    kind: SequenceStepKind
    operands: tuple[str, ...]
    ir: ProductIR | UnaryIR | None
    output: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SequenceIR:
    """Lowered IR for a composed multivector operation.

    A ``SequenceIR`` encodes an ordered list of named computation steps.
    Each step produces a result bound to a symbolic reference (``output``)
    that downstream steps may consume via their ``operands`` tuple.

    This representation lets backends decide between:

    - **Step-by-step execution** (NumPy backend — identical to current
      Python-level composition).
    - **Kernel fusion** (Triton/JAX backends — compile the entire
      sequence into a single GPU kernel).

    Attributes:
        name: Human-readable operation name (``"exp"``, ``"inverse"``, …).
        inputs: Named input references (e.g., ``("mv",)`` or
            ``("actor", "target")``).
        steps: Ordered computation steps.
        result: The named reference holding the final return value.
    """

    name: str
    inputs: tuple[str, ...]
    steps: tuple[IRStep, ...]
    result: str



def build_product_ir(
    plan: OpPlan,
    lhs_storage: StorageKind,
    rhs_storage: StorageKind,
) -> ProductIR:
    """Lower an ``OpPlan`` to storage-aware ``ProductIR``.

    Blade bit-patterns in ``plan.terms`` are translated to layout-local
    column indices so that execution backends can operate directly on
    storage without consulting layout metadata.
    """
    out_blade_to_col = {blade: idx for idx, blade in enumerate(plan.output_blades)}

    terms = tuple(
        TermIR(
            lhs_col=term.lhs_index,
            rhs_col=term.rhs_index,
            out_col=out_blade_to_col[term.out_blade],
            coefficient=term.coefficient,
        )
        for term in plan.terms
    )

    return ProductIR(
        kind=plan.kind,
        lhs_storage=lhs_storage,
        rhs_storage=rhs_storage,
        lhs_width=len(plan.lhs_blades),
        rhs_width=len(plan.rhs_blades),
        out_blades=plan.output_blades,
        terms=terms,
    )


def _reverse_sign(blade: int) -> float:
    """Sign factor for the reverse involution: ``(-1)^{g(g-1)/2}``."""
    g = grade_of_blade(blade)
    return float((-1) ** ((g * (g - 1)) // 2))


def _involute_sign(blade: int) -> float:
    """Sign factor for the grade involution: ``(-1)^g``."""
    return float((-1) ** grade_of_blade(blade))


def build_reverse_ir(blades: tuple[int, ...]) -> UnaryIR:
    """Build ``UnaryIR`` for the reverse operation."""
    return UnaryIR(
        kind="reverse",
        input_width=len(blades),
        out_blades=blades,
        weights=tuple(_reverse_sign(blade) for blade in blades),
    )


def build_involute_ir(blades: tuple[int, ...]) -> UnaryIR:
    """Build ``UnaryIR`` for the involute operation."""
    return UnaryIR(
        kind="involute",
        input_width=len(blades),
        out_blades=blades,
        weights=tuple(_involute_sign(blade) for blade in blades),
    )


def build_conjugate_ir(blades: tuple[int, ...]) -> UnaryIR:
    """Build ``UnaryIR`` for the conjugate operation (reverse ∘ involute)."""
    return UnaryIR(
        kind="conjugate",
        input_width=len(blades),
        out_blades=blades,
        weights=tuple(
            _reverse_sign(blade) * _involute_sign(blade) for blade in blades
        ),
    )


def _pseudoscalar_inverse_scale(algebra: AlgebraSpec) -> float:
    """Return ``1 / (I * I)`` for the metric pseudoscalar."""
    pseudoscalar = algebra.pseudoscalar_blade
    coefficient, _ = algebra.blade_product(pseudoscalar, pseudoscalar)
    if coefficient == 0:
        raise ValueError(
            "dual/undual require an invertible pseudoscalar; this algebra is degenerate."
        )
    return 1.0 / float(coefficient)


def _build_pseudoscalar_ir(
    blades: tuple[int, ...],
    algebra: AlgebraSpec,
    *,
    kind: UnaryKind,
    inverse: bool,
) -> UnaryIR:
    """Build ``UnaryIR`` for metric pseudoscalar dual/undual.

    This mirrors ``ops._pseudoscalar_transform``: each output column is
    read from the input column corresponding to ``target_blade ^ I``,
    weighted by ``source_blade * I`` (scaled by pseudoscalar inverse for
    the dual variant).
    """
    pseudoscalar = algebra.pseudoscalar_blade
    inverse_scale = _pseudoscalar_inverse_scale(algebra) if inverse else 1.0

    source_index = {blade: idx for idx, blade in enumerate(blades)}
    out_blades = tuple(sorted(blade ^ pseudoscalar for blade in blades))

    weights: list[float] = []
    permutation: list[int] = []

    for _out_col, target_blade in enumerate(out_blades):
        source_blade = target_blade ^ pseudoscalar
        source_column = source_index[source_blade]
        coefficient, _ = algebra.blade_product(source_blade, pseudoscalar)
        weights.append(inverse_scale * float(coefficient))
        permutation.append(source_column)

    return UnaryIR(
        kind=kind,
        input_width=len(blades),
        out_blades=out_blades,
        weights=tuple(weights),
        permutation=tuple(permutation),
    )


def build_dual_ir(blades: tuple[int, ...], algebra: AlgebraSpec) -> UnaryIR:
    """Build ``UnaryIR`` for the metric dual operation."""
    return _build_pseudoscalar_ir(blades, algebra, kind="dual", inverse=True)


def build_undual_ir(blades: tuple[int, ...], algebra: AlgebraSpec) -> UnaryIR:
    """Build ``UnaryIR`` for the metric undual operation."""
    return _build_pseudoscalar_ir(blades, algebra, kind="undual", inverse=False)


def _build_poincare_ir(
    blades: tuple[int, ...],
    algebra: AlgebraSpec,
    *,
    kind: UnaryKind,
    inverse: bool,
) -> UnaryIR:
    """Build ``UnaryIR`` for Poincare (metric-free) dual/undual.

    This mirrors ``ops._poincare_transform``: each output column is read
    from the input column corresponding to ``target_blade ^ I``, weighted
    by the appropriate basis product with the pseudoscalar (no inverse
    scaling — pure complement).
    """
    pseudoscalar = algebra.pseudoscalar_blade

    source_index = {blade: idx for idx, blade in enumerate(blades)}
    out_blades = tuple(sorted(blade ^ pseudoscalar for blade in blades))

    weights: list[float] = []
    permutation: list[int] = []

    for _out_col, target_blade in enumerate(out_blades):
        source_blade = target_blade ^ pseudoscalar
        source_column = source_index[source_blade]
        lhs_blade, rhs_blade = (
            (target_blade, source_blade) if inverse else (source_blade, target_blade)
        )
        coefficient, _ = algebra.blade_product(lhs_blade, rhs_blade)
        weights.append(float(coefficient))
        permutation.append(source_column)

    return UnaryIR(
        kind=kind,
        input_width=len(blades),
        out_blades=out_blades,
        weights=tuple(weights),
        permutation=tuple(permutation),
    )


def build_poincare_dual_ir(blades: tuple[int, ...], algebra: AlgebraSpec) -> UnaryIR:
    """Build ``UnaryIR`` for the Poincare dual operation."""
    return _build_poincare_ir(blades, algebra, kind="poincare_dual", inverse=False)


def build_poincare_undual_ir(blades: tuple[int, ...], algebra: AlgebraSpec) -> UnaryIR:
    """Build ``UnaryIR`` for the Poincare undual operation."""
    return _build_poincare_ir(blades, algebra, kind="poincare_undual", inverse=True)


def build_unary_ir(
    blades: tuple[int, ...],
    algebra: AlgebraSpec,
    kind: UnaryKind,
) -> UnaryIR:
    """Dispatch to the correct unary IR builder for *kind*.

    This is the single entry point for ops-layer code: given a layout's
    blade set and the algebra spec, produce the complete unary IR in one
    call.
    """
    if kind == "reverse":
        return build_reverse_ir(blades)
    if kind == "involute":
        return build_involute_ir(blades)
    if kind == "conjugate":
        return build_conjugate_ir(blades)
    if kind == "dual":
        return build_dual_ir(blades, algebra)
    if kind == "undual":
        return build_undual_ir(blades, algebra)
    if kind == "poincare_dual":
        return build_poincare_dual_ir(blades, algebra)
    if kind == "poincare_undual":
        return build_poincare_undual_ir(blades, algebra)
    raise ValueError(f"Unknown UnaryKind: {kind!r}")



def output_layout_from_product_ir(ir: ProductIR, algebra: AlgebraSpec) -> MVLayout:
    """Return the ``MVLayout`` that matches a ``ProductIR``'s output blades."""
    if len(ir.out_blades) == algebra.blade_count:
        return MVLayout.dense(algebra)
    return MVLayout.sparse_pattern(algebra, ir.out_blades, name=_product_ir_layout_name(ir.kind))


def output_layout_from_unary_ir(ir: UnaryIR, algebra: AlgebraSpec) -> MVLayout:
    """Return the ``MVLayout`` that matches a ``UnaryIR``'s output blades."""
    if len(ir.out_blades) == algebra.blade_count:
        return MVLayout.dense(algebra)
    return MVLayout.sparse_pattern(algebra, ir.out_blades, name=ir.kind)


def _product_ir_layout_name(kind: OpKind) -> str:
    return {
        "geometric": "gp",
        "outer": "op",
        "inner": "ip",
        "scalar": "sp",
        "left_contraction": "lc",
        "right_contraction": "rc",
        "regressive": "rp",
    }[kind]


class Executor(Protocol):
    """Execution backend interface for IR-driven operations.

    Each backend must implement three methods that accept IR descriptors
    and return ``MVArray`` results.  The ``MVArray`` type is referenced
    structurally to avoid a hard import cycle — implementers import it
    from ``amsa.mv``.
    """

    def execute_product(
        self,
        lhs: Any,
        rhs: Any,
        ir: ProductIR,
    ) -> Any: ...

    def execute_unary(
        self,
        mv: Any,
        ir: UnaryIR,
    ) -> Any: ...

    def execute_sequence(
        self,
        inputs: dict[str, Any],
        ir: SequenceIR,
    ) -> Any: ...


_BACKENDS: dict[str, Executor] = {}
_DEFAULT_BACKEND: str | None = None


def register_backend(name: str, executor: Executor) -> None:
    """Register an execution backend under the given *name*.

    The first backend registered automatically becomes the default.
    Subsequent registrations do not override the default.

    Examples:

        >>> from amsa.ir import register_backend, MyNumPyExecutor
        >>> register_backend("numpy", MyNumPyExecutor())
    """
    global _DEFAULT_BACKEND
    _BACKENDS[name] = executor
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = name


def unregister_backend(name: str) -> None:
    """Remove a previously registered backend.

    Raises ``KeyError`` if *name* is not registered.
    If the unregistered backend was the default, the default is cleared.
    """
    global _DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise KeyError(f"No backend registered under name {name!r}.")
    del _BACKENDS[name]
    if _DEFAULT_BACKEND == name:
        _DEFAULT_BACKEND = None


def set_default_backend(name: str) -> None:
    """Set the default backend.

    Raises ``KeyError`` if *name* is not registered.
    """
    global _DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise KeyError(f"No backend registered under name {name!r}.")
    _DEFAULT_BACKEND = name


def list_backends() -> tuple[str, ...]:
    """Return the names of all registered backends."""
    return tuple(sorted(_BACKENDS))


def get_backend(name: str | None = None) -> Executor:
    """Resolve an execution backend.

    - If *name* is ``None``, returns the default backend.
    - If *name* is specified, returns the backend registered under that name.

    Raises ``KeyError`` if the requested backend is not found.
    Raises ``RuntimeError`` if no default backend has been registered.
    """
    if name is not None:
        if name not in _BACKENDS:
            raise KeyError(
                f"Backend {name!r} is not registered. "
                f"Available: {list_backends()}."
            )
        return _BACKENDS[name]

    if _DEFAULT_BACKEND is None:
        raise RuntimeError(
            "No default backend configured. "
            "Register a backend with register_backend(name, executor)."
        )
    return _BACKENDS[_DEFAULT_BACKEND]


def has_backend(name: str) -> bool:
    """Check whether a backend is registered under *name*."""
    return name in _BACKENDS


def clear_backends() -> None:
    """Remove all registered backends and clear the default.

    Useful for test isolation.
    """
    global _DEFAULT_BACKEND
    _BACKENDS.clear()
    _DEFAULT_BACKEND = None
