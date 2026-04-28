"""Tests for IR fusion analysis."""

from amsa.fusion import FUSION_PATTERNS, analyze_fusion, apply_fusion_metadata
from amsa.ir import IRStep, SequenceIR
from tests._utils import assert_allclose


def test_fusion_patterns_defined():
    """Test that fusion patterns are defined."""
    assert len(FUSION_PATTERNS) > 0
    pattern_kinds = {p.kind for p in FUSION_PATTERNS}
    assert "scale_product" in pattern_kinds
    assert "unary_product" in pattern_kinds
    assert "elementwise_chain" in pattern_kinds


def test_analyze_fusion_empty_sequence():
    """Test fusion analysis on empty sequence."""
    ir = SequenceIR(name="empty", inputs=(), steps=(), result="output")
    opportunities = analyze_fusion(ir)
    assert opportunities == {}


def test_analyze_fusion_no_match():
    """Test fusion analysis with no matching patterns."""
    ir = SequenceIR(
        name="no_match",
        inputs=("input",),
        steps=(
            IRStep(kind="add", operands=("a", "b"), ir=None, output="c"),
            IRStep(kind="sub", operands=("c", "d"), ir=None, output="e"),
        ),
        result="e",
    )
    opportunities = analyze_fusion(ir)
    assert opportunities == {}


def test_analyze_fusion_scale_product():
    """Test detection of scale + product pattern."""
    ir = SequenceIR(
        name="scale_product",
        inputs=("input",),
        steps=(
            IRStep(
                kind="scale",
                operands=("input",),
                ir=None,
                output="scaled",
                metadata={"factor": 2.0},
            ),
            IRStep(
                kind="binary_product",
                operands=("scaled", "other"),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )
    opportunities = analyze_fusion(ir)
    assert 0 in opportunities
    assert opportunities[0] == "scale_product"


def test_analyze_fusion_unary_product():
    """Test detection of unary + product pattern."""
    ir = SequenceIR(
        name="unary_product",
        inputs=("input",),
        steps=(
            IRStep(kind="unary", operands=("input",), ir=None, output="reversed"),
            IRStep(kind="binary_product", operands=("reversed", "other"), ir=None, output="result"),
        ),
        result="result",
    )
    opportunities = analyze_fusion(ir)
    assert 0 in opportunities
    assert opportunities[0] == "unary_product"


def test_analyze_fusion_elementwise_chain():
    """Test detection of elementwise chain pattern."""
    ir = SequenceIR(
        name="elementwise_chain",
        inputs=("input",),
        steps=(
            IRStep(
                kind="elementwise",
                operands=("input",),
                ir=None,
                output="abs",
                metadata={"function": "abs"},
            ),
            IRStep(
                kind="elementwise",
                operands=("abs",),
                ir=None,
                output="sqrt",
                metadata={"function": "sqrt"},
            ),
        ),
        result="sqrt",
    )
    opportunities = analyze_fusion(ir)
    assert 0 in opportunities
    assert opportunities[0] == "elementwise_chain"


def test_apply_fusion_metadata():
    """Test applying fusion metadata to SequenceIR."""
    ir = SequenceIR(
        name="scale_product",
        inputs=("input",),
        steps=(
            IRStep(
                kind="scale",
                operands=("input",),
                ir=None,
                output="scaled",
                metadata={"factor": 2.0},
            ),
            IRStep(
                kind="binary_product",
                operands=("scaled", "other"),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )
    
    fused_ir = apply_fusion_metadata(ir)
    
    # First step should have fusion metadata
    assert fused_ir.steps[0].fusion == "scale_product"
    # Second step should not have fusion metadata
    assert fused_ir.steps[1].fusion is None


def test_apply_fusion_metadata_no_opportunities():
    """Test applying fusion metadata when no opportunities exist."""
    ir = SequenceIR(
        name="no_match",
        inputs=("input",),
        steps=(
            IRStep(kind="add", operands=("a", "b"), ir=None, output="c"),
            IRStep(kind="sub", operands=("c", "d"), ir=None, output="e"),
        ),
        result="e",
    )
    
    fused_ir = apply_fusion_metadata(ir)
    
    # No steps should have fusion metadata
    for step in fused_ir.steps:
        assert step.fusion is None


def test_fusion_preserves_ir_structure():
    """Test that fusion metadata application preserves IR structure."""
    ir = SequenceIR(
        name="scale_product",
        inputs=("input",),
        steps=(
            IRStep(
                kind="scale",
                operands=("input",),
                ir=None,
                output="scaled",
                metadata={"factor": 2.0},
            ),
            IRStep(
                kind="binary_product",
                operands=("scaled", "other"),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )
    
    fused_ir = apply_fusion_metadata(ir)
    
    # Structure should be preserved
    assert fused_ir.name == ir.name
    assert fused_ir.inputs == ir.inputs
    assert fused_ir.result == ir.result
    assert len(fused_ir.steps) == len(ir.steps)
    
    # Step content should be preserved except for fusion field
    for _i, (orig_step, fused_step) in enumerate(
        zip(ir.steps, fused_ir.steps, strict=True)
    ):
        assert fused_step.kind == orig_step.kind
        assert fused_step.operands == orig_step.operands
        assert fused_step.ir == orig_step.ir
        assert fused_step.output == orig_step.output
        assert fused_step.metadata == orig_step.metadata


def test_fused_scale_product_correctness():
    """Test that fused scale+product produces correct results."""
    import numpy as np

    from amsa.backends.numpy import _execute_fused_scale_product, execute_product_ir
    from amsa.ir import ProductIR, TermIR
    from amsa.layouts import MVLayout
    from amsa.mv import MVArray
    from amsa.specs import AlgebraSpec

    # Create simple test data
    algebra = AlgebraSpec(signature=(1, 1), start_index=1, basis_prefix='e')
    layout = MVLayout.grade(algebra, 1)
    u = MVArray(algebra=algebra, layout=layout, values=np.array([1.0, 2.0]))
    v = MVArray(algebra=algebra, layout=layout, values=np.array([3.0, -4.0]))

    # Create a simple ProductIR (geometric product for grade-1 vectors)
    # For VGA2D: e1*e1 = 1, e1*e2 = e12, e2*e1 = -e12, e2*e2 = -1
    ir = ProductIR(
        kind="geometric",
        lhs_storage="dense",
        rhs_storage="dense",
        lhs_width=2,
        rhs_width=2,
        out_blades=(1, 2, 3),  # scalar, e1, e2
        terms=(
            TermIR(lhs_col=0, rhs_col=0, out_col=0, coefficient=1),  # e1*e1 = 1
            TermIR(
                lhs_col=0,
                rhs_col=1,
                out_col=2,
                coefficient=1,
            ),  # e1*e2 = e12 (not in grade-1 output)
            TermIR(lhs_col=1, rhs_col=0, out_col=2, coefficient=-1),  # e2*e1 = -e12
            TermIR(lhs_col=1, rhs_col=1, out_col=0, coefficient=-1),  # e2*e2 = -1
        ),
    )

    # Non-fused: scale then product
    scaled = MVArray(algebra=algebra, layout=layout, values=u.values * 2.0)
    result_non_fused = execute_product_ir(scaled, v, ir)

    # Fused
    result_fused = _execute_fused_scale_product(u, v, ir, 2.0)

    # Results should match
    assert_allclose(result_fused.values, result_non_fused.values)


def test_fused_elementwise_chain_correctness():
    """Test that fused elementwise chain produces correct results."""
    import numpy as np

    from amsa.backends.numpy import _execute_fused_elementwise_chain

    # Test abs -> sqrt chain
    arr = np.array([-4.0, -9.0, -16.0])

    # Non-fused
    result_non_fused = np.sqrt(np.abs(arr))

    # Fused
    result_fused = _execute_fused_elementwise_chain(
        (arr,),
        ({"function": "abs"}, {"function": "sqrt"}),
    )

    assert_allclose(result_fused, result_non_fused)


def test_fusion_integration_with_backend():
    """Test that fusion metadata is correctly applied and preserved."""
    from amsa.fusion import apply_fusion_metadata
    from amsa.ir import IRStep, SequenceIR

    # Create SequenceIR for scale + binary_product pattern
    sequence_ir = SequenceIR(
        name="scale_product",
        inputs=("u", "v"),
        steps=(
            IRStep(
                kind="scale",
                operands=("u",),
                ir=None,
                output="scaled",
                metadata={"factor": 2.0},
            ),
            IRStep(
                kind="binary_product",
                operands=("scaled", "v"),
                ir=None,
                output="result",
            ),
        ),
        result="result",
    )

    # Apply fusion metadata
    fused_ir = apply_fusion_metadata(sequence_ir)

    # Verify fusion metadata is present
    assert fused_ir.steps[0].fusion == "scale_product"
    assert fused_ir.steps[1].fusion is None

    # Verify IR structure is preserved
    assert fused_ir.name == sequence_ir.name
    assert fused_ir.inputs == sequence_ir.inputs
    assert fused_ir.result == sequence_ir.result
    assert len(fused_ir.steps) == len(sequence_ir.steps)


def test_fusion_unary_product_integration():
    """Test that unary + product fusion detection works (execution tested separately)."""
    from amsa.fusion import apply_fusion_metadata
    from amsa.ir import IRStep, SequenceIR

    # Create SequenceIR for unary + product pattern
    sequence_ir = SequenceIR(
        name="unary_product",
        inputs=("u", "v"),
        steps=(
            IRStep(kind="unary", operands=("u",), ir=None, output="reversed"),
            IRStep(kind="binary_product", operands=("reversed", "v"), ir=None, output="result"),
        ),
        result="result",
    )

    # Apply fusion metadata
    fused_ir = apply_fusion_metadata(sequence_ir)

    # Verify fusion metadata is present
    assert fused_ir.steps[0].fusion == "unary_product"


def test_fusion_elementwise_chain_integration():
    """Test that elementwise chain fusion metadata is correctly applied."""
    from amsa.fusion import apply_fusion_metadata
    from amsa.ir import IRStep, SequenceIR

    # Create SequenceIR for elementwise chain (abs -> sqrt)
    sequence_ir = SequenceIR(
        name="elementwise_chain",
        inputs=("input",),
        steps=(
            IRStep(
                kind="elementwise",
                operands=("input",),
                ir=None,
                output="abs",
                metadata={"function": "abs"},
            ),
            IRStep(
                kind="elementwise",
                operands=("abs",),
                ir=None,
                output="sqrt",
                metadata={"function": "sqrt"},
            ),
        ),
        result="sqrt",
    )

    # Apply fusion metadata
    fused_ir = apply_fusion_metadata(sequence_ir)

    # Verify fusion metadata is present
    assert fused_ir.steps[0].fusion == "elementwise_chain"
    assert fused_ir.steps[1].fusion is None


def test_fusion_no_opportunity_unchanged():
    """Test that non-fusible sequences have no fusion metadata."""
    from amsa.fusion import apply_fusion_metadata
    from amsa.ir import IRStep, SequenceIR

    # Create SequenceIR with non-fusible pattern (add -> sub)
    sequence_ir = SequenceIR(
        name="non_fusible",
        inputs=("a", "b"),
        steps=(
            IRStep(kind="add", operands=("a", "b"), ir=None, output="sum"),
            IRStep(kind="sub", operands=("sum", "a"), ir=None, output="result"),
        ),
        result="result",
    )

    # Apply fusion metadata (should find no opportunities)
    fused_ir = apply_fusion_metadata(sequence_ir)

    # Verify no fusion metadata
    for step in fused_ir.steps:
        assert step.fusion is None
