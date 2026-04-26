# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for IR fusion analysis."""

from amsa.fusion import analyze_fusion, apply_fusion_metadata, FUSION_PATTERNS
from amsa.ir import IRStep, SequenceIR


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
            IRStep(kind="scale", operands=("input",), ir=None, output="scaled", metadata={"factor": 2.0}),
            IRStep(kind="binary_product", operands=("scaled", "other"), ir=None, output="result"),
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
            IRStep(kind="elementwise", operands=("input",), ir=None, output="abs", metadata={"function": "abs"}),
            IRStep(kind="elementwise", operands=("abs",), ir=None, output="sqrt", metadata={"function": "sqrt"}),
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
            IRStep(kind="scale", operands=("input",), ir=None, output="scaled", metadata={"factor": 2.0}),
            IRStep(kind="binary_product", operands=("scaled", "other"), ir=None, output="result"),
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
            IRStep(kind="scale", operands=("input",), ir=None, output="scaled", metadata={"factor": 2.0}),
            IRStep(kind="binary_product", operands=("scaled", "other"), ir=None, output="result"),
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
    for i, (orig_step, fused_step) in enumerate(zip(ir.steps, fused_ir.steps)):
        assert fused_step.kind == orig_step.kind
        assert fused_step.operands == orig_step.operands
        assert fused_step.ir == orig_step.ir
        assert fused_step.output == orig_step.output
        assert fused_step.metadata == orig_step.metadata
