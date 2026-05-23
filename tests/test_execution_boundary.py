from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "amsa"


BOUNDARY_CLASSIFICATION = {
    "src/amsa/__init__.py": "package wiring",
    "src/amsa/algebra.py": "construction/coercion pending backend routing",
    "src/amsa/autodiff.py": "reference forward-mode autodiff",
    "src/amsa/backends/__init__.py": "backend namespace",
    "src/amsa/backends/jax.py": "backend execution",
    "src/amsa/backends/numpy.py": "backend execution",
    "src/amsa/cga.py": "CGA domain constructors",
    "src/amsa/fusion.py": "IR fusion analysis",
    "src/amsa/inspection.py": "geometric classification and inspection",
    "src/amsa/ir.py": "IR metadata and backend registry",
    "src/amsa/layouts.py": "layout metadata",
    "src/amsa/mv.py": "container/coercion pending backend routing",
    "src/amsa/ops.py": "public ops pending fuller IR routing",
    "src/amsa/plans.py": "product planning",
    "src/amsa/specs.py": "algebra semantics",
    "src/amsa/storage.py": "storage execution pending backend routing",
}

PENDING_EXECUTION_ROUTING = {
    "src/amsa/algebra.py",
    "src/amsa/mv.py",
    "src/amsa/ops.py",
    "src/amsa/storage.py",
}

EXPECTED_NUMPY_CORE_FILES = {
    "src/amsa/algebra.py",
    "src/amsa/autodiff.py",
    "src/amsa/backends/numpy.py",
    "src/amsa/cga.py",
    "src/amsa/inspection.py",
    "src/amsa/mv.py",
    "src/amsa/ops.py",
    "src/amsa/specs.py",
    "src/amsa/storage.py",
}

EXPECTED_LOOP_CORE_FILES = {
    "src/amsa/algebra.py",
    "src/amsa/autodiff.py",
    "src/amsa/backends/jax.py",
    "src/amsa/backends/numpy.py",
    "src/amsa/cga.py",
    "src/amsa/fusion.py",
    "src/amsa/inspection.py",
    "src/amsa/ir.py",
    "src/amsa/layouts.py",
    "src/amsa/mv.py",
    "src/amsa/ops.py",
    "src/amsa/plans.py",
    "src/amsa/specs.py",
    "src/amsa/storage.py",
}


@dataclass(frozen=True)
class BoundarySite:
    path: str
    line: int
    kind: str


def _core_python_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in SRC.rglob("*.py")
            if "viz" not in path.relative_to(SRC).parts
        )
    )


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _numpy_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    aliases.add(alias.asname or "numpy")
    return aliases


def _scan_boundary_sites() -> tuple[BoundarySite, ...]:
    sites: list[BoundarySite] = []
    for path in _core_python_files():
        rel = _relative(path)
        tree = ast.parse(path.read_text(), filename=rel)
        numpy_aliases = _numpy_aliases(tree)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in numpy_aliases
            ):
                sites.append(BoundarySite(rel, node.lineno, f"{node.value.id}.{node.attr}"))
            elif isinstance(node, ast.For):
                sites.append(BoundarySite(rel, node.lineno, "for"))
            elif isinstance(node, ast.While):
                sites.append(BoundarySite(rel, node.lineno, "while"))
    return tuple(sites)


def test_core_execution_boundary_inventory_is_classified() -> None:
    sites = _scan_boundary_sites()
    files_with_sites = {site.path for site in sites}

    unclassified = sorted(files_with_sites - set(BOUNDARY_CLASSIFICATION))

    assert unclassified == []
    assert PENDING_EXECUTION_ROUTING <= files_with_sites


def test_numpy_usage_inventory_matches_pass1_audit() -> None:
    sites = _scan_boundary_sites()
    numpy_files = {site.path for site in sites if site.kind.startswith(("np.", "numpy."))}

    assert numpy_files == EXPECTED_NUMPY_CORE_FILES


def test_loop_inventory_matches_pass1_audit() -> None:
    sites = _scan_boundary_sites()
    loop_files = {site.path for site in sites if site.kind in {"for", "while"}}

    assert loop_files == EXPECTED_LOOP_CORE_FILES


def test_pass1_pending_execution_debt_is_explicitly_named() -> None:
    assert BOUNDARY_CLASSIFICATION["src/amsa/ops.py"] == "public ops pending fuller IR routing"
    assert (
        BOUNDARY_CLASSIFICATION["src/amsa/storage.py"]
        == "storage execution pending backend routing"
    )
    assert (
        BOUNDARY_CLASSIFICATION["src/amsa/algebra.py"]
        == "construction/coercion pending backend routing"
    )
    assert (
        BOUNDARY_CLASSIFICATION["src/amsa/mv.py"]
        == "container/coercion pending backend routing"
    )
