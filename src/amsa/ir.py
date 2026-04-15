"""Intermediate representation for AMSA execution backends.

This module defines storage-aware, backend-neutral IR descriptors that
capture the computation graph of Clifford algebra operations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from amsa.plans import OpKind
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
