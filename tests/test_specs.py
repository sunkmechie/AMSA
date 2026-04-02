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


def test_blade_product_respects_metric_and_sign() -> None:
    spec = vga(3)
    assert spec.blade_product(spec.blade_from_key("e1"), spec.blade_from_key("e1")) == (1, 0)
    assert spec.blade_product(spec.blade_from_key("e2"), spec.blade_from_key("e1")) == (-1, 0b11)

    pga = pga2d()
    assert pga.blade_product(pga.blade_from_key("e0"), pga.blade_from_key("e0")) == (0, 0)


def test_basis_product_table_matches_pre_table_blade_products() -> None:
    spec = vga(3)
    expected = {
        (lhs, rhs): spec.blade_product(lhs, rhs)
        for lhs in range(spec.blade_count)
        for rhs in range(spec.blade_count)
    }

    table = spec.basis_product_table

    assert table is not None
    assert table.blade_count == spec.blade_count
    assert tuple(int(grade) for grade in table.grades) == spec.grades_of_blades()
    for pair, product in expected.items():
        assert table.blade_product(*pair) == product


def test_cayley_table_uses_canonical_names_and_zero_entries() -> None:
    spec = pga2d()
    cayley = spec.cayley_table()

    assert cayley[("e0", "e0")] == "0"
    assert cayley[("e1", "e2")] == "e12"
    assert cayley[("e2", "e1")] == "-e12"


def test_large_algebra_skips_precomputed_basis_product_table() -> None:
    spec = vga(10)

    assert spec.basis_product_table is None
