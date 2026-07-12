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

"""IR fusion analysis for AMSA.

This module defines fusion patterns and analysis passes for optimizing
SequenceIR by combining adjacent operations into fused kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from amsa.ir import IRStep, SequenceIR, SequenceStepKind

FusionKind = Literal[
    "scale_product",  # scale followed by binary product
    "unary_product",  # unary followed by binary product
]


@dataclass(frozen=True, slots=True)
class FusionPattern:
    """A fusion pattern describing how operations can be combined.

    Attributes:
        kind: The type of fusion.
        step_kinds: The sequence of step kinds that match this pattern.
        requires_metadata: Optional metadata keys that must be present.
    """

    kind: FusionKind
    step_kinds: tuple[SequenceStepKind, ...]
    requires_metadata: tuple[str, ...] = ()


FUSION_PATTERNS: tuple[FusionPattern, ...] = (
    FusionPattern(
        kind="scale_product",
        step_kinds=("scale", "binary_product"),
    ),
    FusionPattern(
        kind="unary_product",
        step_kinds=("unary", "binary_product"),
    ),
)


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_metadata(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_metadata(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _cse_step_key(step: IRStep) -> tuple[Any, ...]:
    return (
        step.kind,
        step.operands,
        step.ir,
        _freeze_metadata(step.metadata),
    )


def eliminate_common_subexpressions(ir: SequenceIR) -> SequenceIR:
    """Remove duplicate pure steps from a ``SequenceIR``.

    The pass is intentionally conservative: a step is common only when its kind,
    remapped operands, lowered IR, and metadata are identical. Sequence steps are
    pure coefficient operations, so later duplicate outputs can safely alias the
    first output.
    """
    output_aliases: dict[str, str] = {}
    seen: dict[tuple[Any, ...], str] = {}
    new_steps: list[IRStep] = []

    for step in ir.steps:
        operands = tuple(output_aliases.get(operand, operand) for operand in step.operands)
        remapped = IRStep(
            kind=step.kind,
            operands=operands,
            ir=step.ir,
            output=step.output,
            metadata=step.metadata,
            fusion=None,
        )
        key = _cse_step_key(remapped)
        existing_output = seen.get(key)
        if existing_output is not None:
            output_aliases[step.output] = existing_output
            continue

        seen[key] = step.output
        output_aliases[step.output] = step.output
        new_steps.append(remapped)

    return SequenceIR(
        name=ir.name,
        inputs=ir.inputs,
        steps=tuple(new_steps),
        result=output_aliases.get(ir.result, ir.result),
    )


def optimize_sequence_ir(ir: SequenceIR) -> SequenceIR:
    """Apply conservative sequence optimizations used by eager backends."""
    return apply_fusion_metadata(eliminate_common_subexpressions(ir))


def analyze_fusion(ir: SequenceIR) -> dict[int, FusionKind]:
    """Analyze a SequenceIR and identify fusion opportunities.

    Args:
        ir: The SequenceIR to analyze.

    Returns:
        A mapping from step index to fusion kind for steps that can be fused.
        The index refers to the first step in the fusible sequence.
    """
    fusion_opportunities: dict[int, FusionKind] = {}

    for pattern in FUSION_PATTERNS:
        # Scan for pattern matches
        for i in range(len(ir.steps) - len(pattern.step_kinds) + 1):
            match = True
            for j, expected_kind in enumerate(pattern.step_kinds):
                if ir.steps[i + j].kind != expected_kind:
                    match = False
                    break

            if match:
                # Check metadata requirements
                metadata_ok = True
                for key in pattern.requires_metadata:
                    step_metadata = ir.steps[i].metadata
                    if step_metadata is None or key not in step_metadata:
                        metadata_ok = False
                        break

                if metadata_ok:
                    fusion_opportunities[i] = pattern.kind

    return fusion_opportunities


def apply_fusion_metadata(ir: SequenceIR) -> SequenceIR:
    """Apply fusion metadata to a SequenceIR based on analysis.

    Args:
        ir: The SequenceIR to annotate with fusion metadata.

    Returns:
        A new SequenceIR with fusion metadata applied to fusible steps.
    """
    fusion_opportunities = analyze_fusion(ir)

    # Rebuild steps with fusion metadata
    new_steps = []
    for i, step in enumerate(ir.steps):
        if i in fusion_opportunities:
            # Mark the first step of the fusible sequence
            new_step = IRStep(
                kind=step.kind,
                operands=step.operands,
                ir=step.ir,
                output=step.output,
                metadata=step.metadata,
                fusion=fusion_opportunities[i],
            )
            new_steps.append(new_step)
        else:
            new_steps.append(step)

    return SequenceIR(
        name=ir.name,
        inputs=ir.inputs,
        steps=tuple(new_steps),
        result=ir.result,
    )
