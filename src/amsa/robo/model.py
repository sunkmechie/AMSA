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

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Link:
    name: str


@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    child_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    child_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RobotModel:
    name: str
    links: tuple[Link, ...] = ()
    joints: tuple[Joint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _triple(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    values = tuple(float(part) for part in text.split())
    if len(values) != 3:
        raise ValueError("Expected three floating point values.")
    return values


def importurdf(path: str | Path) -> RobotModel:
    """Import URDF topology into the experimental robotics model.

    This is intentionally a topology/metadata bridge, not AMSA's native robot
    representation.
    """
    root = ET.parse(path).getroot()
    links = tuple(Link(node.attrib["name"]) for node in root.findall("link"))
    joints: list[Joint] = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        parent = node.find("parent")
        child = node.find("child")
        if parent is None or child is None:
            joint_name = node.attrib.get("name", "<unnamed>")
            raise ValueError(f"URDF joint {joint_name} is missing parent/child.")
        joints.append(
            Joint(
                name=node.attrib["name"],
                kind=node.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                axis=_triple(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                origin_xyz=_triple(
                    origin.attrib.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                origin_rpy=_triple(
                    origin.attrib.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                motion=_default_joint_motion(
                    node.attrib.get("type", "fixed"),
                    _triple(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                ),
            )
        )
    return RobotModel(
        name=root.attrib.get("name", Path(path).stem),
        links=links,
        joints=tuple(joints),
    )


def load(path: str | Path, *, type: str | None = None) -> RobotModel:
    """Load a robot model and normalize it to AMSA's draft ``RobotModel``.

    Supported formats are ``"urdf"`` and ``"crobot"``.  When ``type`` is not
    provided, the format is inferred from the file extension.
    """
    resolved_type = type or Path(path).suffix.lstrip(".").lower()
    if resolved_type == "urdf":
        return importurdf(path)
    if resolved_type == "crobot":
        return load_crobot(path)
    raise ValueError(f"Unsupported robot model type: {resolved_type!r}.")


def load_crobot(path: str | Path) -> RobotModel:
    """Load AMSA's experimental Clifford-native robot JSON format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RobotModel(
        name=data["name"],
        links=tuple(Link(item["name"]) for item in data.get("links", [])),
        joints=tuple(
            Joint(
                name=item["name"],
                kind=item["kind"],
                parent=item["parent"],
                child=item["child"],
                axis=tuple(item.get("axis", (0.0, 0.0, 1.0))),
                origin_xyz=tuple(item.get("origin_xyz", (0.0, 0.0, 0.0))),
                origin_rpy=tuple(item.get("origin_rpy", (0.0, 0.0, 0.0))),
                child_offset_xyz=tuple(item.get("child_offset_xyz", (0.0, 0.0, 0.0))),
                child_offset_rpy=tuple(item.get("child_offset_rpy", (0.0, 0.0, 0.0))),
                motion=_load_joint_motion(item),
            )
            for item in data.get("joints", [])
        ),
        metadata=dict(data.get("metadata", {})),
    )


def dump_crobot(model: RobotModel) -> dict[str, Any]:
    """Return a serializable draft `.crobot` document."""
    return {
        "format": "amsa-crobot",
        "version": 0,
        "name": model.name,
        "model": "cga3d",
        "links": [{"name": link.name} for link in model.links],
        "joints": [
            {
                "name": joint.name,
                "kind": joint.kind,
                "parent": joint.parent,
                "child": joint.child,
                "axis": joint.axis,
                "origin_xyz": joint.origin_xyz,
                "origin_rpy": joint.origin_rpy,
                "child_offset_xyz": joint.child_offset_xyz,
                "child_offset_rpy": joint.child_offset_rpy,
                "motion": _joint_motion(joint),
            }
            for joint in model.joints
        ],
        "metadata": dict(model.metadata),
    }


def _load_joint_motion(item: dict[str, Any]) -> dict[str, Any]:
    motion = item.get("motion")
    if isinstance(motion, dict):
        return dict(motion)
    return _default_joint_motion(item.get("kind", "fixed"), item.get("axis", (0.0, 0.0, 1.0)))


def _joint_motion(joint: Joint) -> dict[str, Any]:
    if joint.motion:
        return dict(joint.motion)
    return _default_joint_motion(joint.kind, joint.axis)


def _default_joint_motion(kind: str, axis: Any) -> dict[str, Any]:
    axis_tuple = tuple(float(item) for item in axis)
    if kind == "fixed":
        return {
            "model": "cga3d",
            "kind": "fixed",
            "parameter": "none",
            "generator": {"kind": "identity"},
        }
    if kind == "prismatic":
        return {
            "model": "cga3d",
            "kind": "prismatic",
            "parameter": "distance",
            "generator": {
                "kind": "translation-direction",
                "axis": axis_tuple,
            },
        }
    if kind == "revolute":
        return {
            "model": "cga3d",
            "kind": "revolute",
            "parameter": "angle",
            "generator": {
                "kind": "rotation-axis",
                "axis": axis_tuple,
            },
        }
    return {
        "model": "cga3d",
        "kind": kind,
        "parameter": "unknown",
        "generator": {"kind": "unsupported", "axis": axis_tuple},
    }


def model_from_dh(
    dh_params: list[tuple[float, float, float, float]],
    *,
    joint_types: list[str] | None = None,
    name: str = "dh_chain",
) -> RobotModel:
    """Build an executable draft ``RobotModel`` from DH parameters.

    The resulting model stores each variable joint as a CGA generator axis plus
    fixed child offsets.  DH is treated as an adapter shape, not as the native
    execution representation.
    """
    from amsa.robo._validation import _validate_joint_types

    n = len(dh_params)
    joint_types = _validate_joint_types(n, joint_types)
    links = [Link("base")]
    joints: list[Joint] = []
    for i, ((alpha, a, d, theta), kind) in enumerate(zip(dh_params, joint_types, strict=True)):
        parent = links[-1].name
        child = f"link_{i + 1}"
        links.append(Link(child))
        joints.append(
            Joint(
                name=f"joint_{i + 1}",
                kind=kind,
                parent=parent,
                child=child,
                axis=(0.0, 0.0, 1.0),
                origin_xyz=(0.0, 0.0, float(d)),
                origin_rpy=(0.0, 0.0, float(theta) if kind == "prismatic" else 0.0),
                child_offset_xyz=(float(a), 0.0, 0.0),
                child_offset_rpy=(float(alpha), 0.0, 0.0),
                motion=_default_joint_motion(kind, (0.0, 0.0, 1.0)),
            )
        )
    return RobotModel(
        name=name,
        links=tuple(links),
        joints=tuple(joints),
        metadata={"source": "dh", "format": "amsa-crobot-draft"},
    )


def active_joints(model: RobotModel) -> tuple[Joint, ...]:
    """Return joints that consume a model-level joint parameter."""
    return tuple(joint for joint in model.joints if _joint_motion(joint).get("kind") != "fixed")


def serial_chain(model: RobotModel, base_link: str, tip_link: str) -> RobotModel:
    """Extract a single parent-to-child joint path from a loaded robot model."""
    by_child = {joint.child: joint for joint in model.joints}
    chain: list[Joint] = []
    current = tip_link
    while current != base_link:
        joint = by_child.get(current)
        if joint is None:
            raise ValueError(f"No joint path from {base_link!r} to {tip_link!r}.")
        chain.append(joint)
        current = joint.parent
    chain.reverse()

    link_names = [base_link]
    link_names.extend(joint.child for joint in chain)
    available_links = {link.name for link in model.links}
    links = tuple(Link(name) for name in link_names if name in available_links)
    return RobotModel(
        name=f"{model.name}:{base_link}->{tip_link}",
        links=links,
        joints=tuple(chain),
        metadata={**model.metadata, "source_model": model.name},
    )


def _resolve_model_joint_values(
    model: RobotModel,
    joint_values: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    values = np.asarray(joint_values, dtype=float)
    active_count = len(active_joints(model))
    if values.shape == (len(model.joints),):
        return values
    if values.shape != (active_count,):
        raise ValueError(
            f"Expected {active_count} active joint values or {len(model.joints)} total joint "
            f"values, got shape {values.shape}."
        )

    expanded = np.zeros(len(model.joints), dtype=float)
    cursor = 0
    for i, joint in enumerate(model.joints):
        if _joint_motion(joint).get("kind") == "fixed":
            continue
        expanded[i] = values[cursor]
        cursor += 1
    return expanded
