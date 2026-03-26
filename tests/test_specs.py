import pytest

from amsa import Algebra, AlgebraSpec, pga2d, pga3d, vga
from amsa.specs import canonical_blade_name, grade_of_blade


def test_grade_of_blade_uses_bit_count() -> None:
    assert grade_of_blade(0b0000) == 0
    assert grade_of_blade(0b0101) == 2
    assert grade_of_blade(0b1111) == 4


def test_grade_of_blade_rejects_negative_input() -> None:
    with pytest.raises(ValueError):
        grade_of_blade(-1)


def test_canonical_blade_name_uses_basis_indices() -> None:
    assert canonical_blade_name(0, dimension=3) == "e"
    assert canonical_blade_name(1, dimension=3) == "e1"
    assert canonical_blade_name(0b101, dimension=3) == "e13"


def test_algebra_spec_normalizes_pqr() -> None:
    spec = AlgebraSpec.from_pqr(3, 1, 1)
    assert spec.signature == (0, 1, 1, 1, -1)
    assert spec.p == 3
    assert spec.q == 1
    assert spec.r == 1


def test_vga_basis_generation() -> None:
    spec = vga(3)
    assert spec.dimension == 3
    assert spec.blade_count == 8
    assert spec.blades_of_grade(1) == (1, 2, 4)
    assert spec.blade_name(7) == "e123"


def test_named_presets_match_expected_signatures() -> None:
    assert pga2d().signature == (0, 1, 1)
    assert pga3d().signature == (0, 1, 1, 1)


def test_algebra_wrapper_exposes_layout_constructors() -> None:
    algebra = Algebra(vga(2))
    assert algebra.signature == (1, 1)
    assert algebra.dense_layout().size == 4
