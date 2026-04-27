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
from typing import Literal

from amsa.ir import SequenceIR, SequenceStepKind

FusionKind = Literal[
    "scale_product",  # scale followed by binary product
    "unary_product",  # unary followed by binary product
    "sequential_products",  # chain of binary products
    "elementwise_chain",  # chain of elementwise operations
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


# Define fusion patterns
FUSION_PATTERNS: tuple[FusionPattern, ...] = (
    # Scale + product: scale followed by binary_product
    FusionPattern(
        kind="scale_product",
        step_kinds=("scale", "binary_product"),
    ),
    # Unary + product: unary followed by binary_product
    FusionPattern(
        kind="unary_product",
        step_kinds=("unary", "binary_product"),
    ),
    # Elementwise chains: multiple elementwise operations
    FusionPattern(
        kind="elementwise_chain",
        step_kinds=("elementwise", "elementwise"),
    ),
)


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
    from amsa.ir import IRStep

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
