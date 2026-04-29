"""Tests for inspection and pretty-print API."""


from amsa import Algebra
from amsa.ir import build_product_ir
from amsa.plans import plan_binary_product


def test_mvarray_repr_simple() -> None:
    """Test MVArray.__repr__ for simple multivector."""
    alg = Algebra.vga2d()
    mv = alg.vector([1.0, 2.0])
    repr_str = repr(mv)
    assert "e1" in repr_str or "e2" in repr_str
    assert "1.0" in repr_str or "2.0" in repr_str


def test_mvarray_repr_zero() -> None:
    """Test MVArray.__repr__ for zero multivector."""
    alg = Algebra.vga2d()
    mv = alg.zeros()
    repr_str = repr(mv)
    assert repr_str == "0"


def test_mvarray_repr_bivector() -> None:
    """Test MVArray.__repr__ for bivector."""
    alg = Algebra.vga2d()
    mv = alg.bivector([3.0])
    repr_str = repr(mv)
    assert "e12" in repr_str
    assert "3.0" in repr_str


def test_mvarray_repr_batched() -> None:
    """Test MVArray.__repr__ for batched multivector."""
    alg = Algebra.vga2d()
    mv = alg.zeros(batch_shape=(2, 3))
    repr_str = repr(mv)
    assert "batch_shape=(2, 3)" in repr_str
    assert "blades=" in repr_str
    assert "dtype=" in repr_str


def test_mvarray_repr_mixed_grades() -> None:
    """Test MVArray.__repr__ for mixed-grade multivector."""
    alg = Algebra.vga2d()
    mv = alg.multivector({0: 1.0, 1: 2.0, 3: 3.0})
    repr_str = repr(mv)
    assert "e" in repr_str or "e1" in repr_str or "e12" in repr_str


def test_opplan_show() -> None:
    """Test OpPlan.show() displays product plan in readable format."""
    alg = Algebra.vga2d()
    lhs_layout = alg.grade_layout(1)
    rhs_layout = alg.grade_layout(1)
    plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")
    
    show_str = plan.show()
    assert "OpPlan(geometric)" in show_str
    assert "LHS blades:" in show_str
    assert "RHS blades:" in show_str
    assert "Output blades:" in show_str
    assert "Terms" in show_str


def test_productir_show() -> None:
    """Test ProductIR.show() displays IR in readable format."""
    alg = Algebra.vga2d()
    lhs_layout = alg.grade_layout(1)
    rhs_layout = alg.grade_layout(1)
    plan = plan_binary_product(lhs_layout, rhs_layout, "geometric")
    ir = build_product_ir(plan, "dense", "dense")
    
    show_str = ir.show(alg.spec)
    assert "ProductIR(geometric)" in show_str
    assert "LHS storage:" in show_str
    assert "RHS storage:" in show_str
    assert "Output blades:" in show_str
    assert "Terms" in show_str
    assert "col[" in show_str


def test_algebra_show_cayley() -> None:
    """Test Algebra.show_cayley() displays Cayley table subset."""
    alg = Algebra.vga2d()
    show_str = alg.show_cayley()
    assert "Cayley table" in show_str
    assert "e" in show_str or "e1" in show_str


def test_algebra_show_cayley_custom_blades() -> None:
    """Test Algebra.show_cayley() with custom blade selection."""
    alg = Algebra.vga2d()
    show_str = alg.show_cayley(blades=(0, 1, 2))
    assert "Cayley table" in show_str
    # Should only show 3 blades
