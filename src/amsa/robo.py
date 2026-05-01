# Copyright 2026 Surya Sunkara
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENTAL_WARNING = (
    "amsa.robo is experimental and not ready for production robotics use. "
    "APIs and file formats may change before amsa-robo is split out."
)


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
            )
        )
    return RobotModel(
        name=root.attrib.get("name", Path(path).stem),
        links=links,
        joints=tuple(joints),
    )


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
                "motion": "bivector-generator",
            }
            for joint in model.joints
        ],
        "metadata": dict(model.metadata),
    }


def ik(*args: Any, **kwargs: Any) -> Any:
    """Experimental inverse-kinematics namespace entry point."""
    if kwargs.get("solver") == "planar_two_link" and len(args) >= 2:
        return planar_two_link_ik(args[0], args[1], elbow=kwargs.get("elbow", "up"))
    raise NotImplementedError(
        "amsa.robo.ik is experimental; currently use solver='planar_two_link'."
    )


def planar_two_link_ik(
    link_lengths: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    elbow: str = "up",
) -> tuple[float, float]:
    """Minimal analytic 2R IK helper for smoke-testing the robotics namespace."""
    l1, l2 = link_lengths
    x, y = target_xy
    cos_q2 = np.clip((x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    sin_q2 = math.sqrt(max(0.0, 1.0 - float(cos_q2 * cos_q2)))
    if elbow == "down":
        sin_q2 = -sin_q2
    q2 = math.atan2(sin_q2, float(cos_q2))
    q1 = math.atan2(y, x) - math.atan2(l2 * sin_q2, l1 + l2 * float(cos_q2))
    return q1, q2


__all__ = [
    "EXPERIMENTAL_WARNING",
    "Joint",
    "Link",
    "RobotModel",
    "dump_crobot",
    "ik",
    "importurdf",
    "load_crobot",
    "planar_two_link_ik",
]
