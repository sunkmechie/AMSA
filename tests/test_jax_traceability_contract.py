from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATIC_TRACE_METADATA = frozenset(
    {
        "AlgebraSpec",
        "MVLayout",
        "ProductIR",
        "UnaryIR",
        "storage kind",
        "width",
    }
)

DYNAMIC_TRACE_VALUES = frozenset(
    {
        "coefficient arrays",
        "scalar coefficient inputs",
        "batch contents",
    }
)

SUPPORTED_DENSE_TRACE_TARGETS = frozenset(
    {
        "geometric",
        "outer",
        "inner",
        "scalar",
        "left contraction",
        "right contraction",
        "regressive product",
        "reverse",
        "involute",
        "conjugate",
        "Poincare dual",
        "Poincare undual",
        "add",
        "sub",
        "scale",
        "row_scale",
        "grade projection",
        "sandwich",
        "norm_squared",
        "scalar-objective autodiff",
    }
)

DEFERRED_TRACE_TARGETS = frozenset(
    {
        "CSR storage on JAX",
        "value-dependent output support",
        "value-dependent output shapes",
        "Python exceptions triggered from traced coefficient values",
        "singular normalization branches inside jax.jit",
        "predicate helpers that intentionally return Python bool values",
        "normalize",
        "inverse",
    }
)


def test_traceability_contract_has_disjoint_static_and_dynamic_roles() -> None:
    assert STATIC_TRACE_METADATA
    assert DYNAMIC_TRACE_VALUES
    assert STATIC_TRACE_METADATA.isdisjoint(DYNAMIC_TRACE_VALUES)


def test_dense_trace_targets_are_explicitly_named() -> None:
    required_targets = {
        "geometric",
        "outer",
        "inner",
        "regressive product",
        "reverse",
        "Poincare dual",
        "add",
        "row_scale",
        "sandwich",
        "norm_squared",
        "scalar-objective autodiff",
    }

    assert required_targets <= SUPPORTED_DENSE_TRACE_TARGETS


def test_deferred_trace_targets_are_explicitly_named() -> None:
    required_deferred_targets = {
        "CSR storage on JAX",
        "value-dependent output shapes",
        "Python exceptions triggered from traced coefficient values",
        "singular normalization branches inside jax.jit",
        "normalize",
        "inverse",
    }

    assert required_deferred_targets <= DEFERRED_TRACE_TARGETS
    assert SUPPORTED_DENSE_TRACE_TARGETS.isdisjoint(DEFERRED_TRACE_TARGETS)


def test_backend_docs_include_traceability_contract_terms() -> None:
    docs = (ROOT / "docs" / "backends.rst").read_text()
    normalized_docs = docs.replace("``", "")

    for term in STATIC_TRACE_METADATA | DYNAMIC_TRACE_VALUES:
        assert term in normalized_docs

    for target in SUPPORTED_DENSE_TRACE_TARGETS:
        assert target in normalized_docs

    for target in DEFERRED_TRACE_TARGETS:
        assert target in normalized_docs
