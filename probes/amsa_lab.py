from __future__ import annotations

import argparse
import html
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import amsa
import amsa.ops as ops_module
from amsa import Algebra, MVArray
from amsa.plans import OpKind, OpPlan

DEFAULT_ALGEBRA = "vga2d"
DEFAULT_STATEMENTS = (
    "u = alg.vector([1.0, 2.0])",
    "v = alg.vector([3.0, 4.0])",
)
DEFAULT_EXPRESSION = "u * v"
DEFAULT_OUTPUT = Path("tempo/amsa_lab.html")

OPERATOR_SYMBOLS: dict[OpKind, str] = {
    "geometric": "*",
    "outer": "^",
    "inner": "|",
    "left_contraction": "left_contract",
    "right_contraction": "right_contract",
    "regressive": "regress",
}


@dataclass(frozen=True, slots=True)
class TermContribution:
    lhs_blade: int
    rhs_blade: int
    out_blade: int
    coefficient: int
    lhs_value: Any
    rhs_value: Any
    value: Any


@dataclass(frozen=True, slots=True)
class ProductTrace:
    kind: OpKind
    operator_symbol: str
    plan: OpPlan
    lhs_text: str
    rhs_text: str
    result_text: str
    batch_shape: tuple[int, ...]
    sample_index: tuple[int, ...]
    lhs_values: tuple[Any, ...]
    rhs_values: tuple[Any, ...]
    result_values: tuple[Any, ...]
    contributions: tuple[TermContribution, ...]


@dataclass(frozen=True, slots=True)
class TraceReport:
    algebra_name: str
    expression: str
    statements: tuple[str, ...]
    final_result_text: str
    traces: tuple[ProductTrace, ...]


def _normalize_scalar(value: Any) -> Any:
    scalar = np.asarray(value).item()
    if isinstance(scalar, complex):
        if np.isclose(scalar.imag, 0.0):
            scalar = float(scalar.real)
        else:
            return scalar

    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    if isinstance(scalar, (np.floating, float)):
        number = float(scalar)
        rounded = round(number)
        if np.isclose(number, rounded):
            return int(rounded)
        return number
    return scalar


def _is_zero(value: Any) -> bool:
    return bool(np.isclose(np.asarray(value), 0.0).item())


def _format_scalar(value: Any) -> str:
    scalar = _normalize_scalar(value)
    if isinstance(scalar, complex):
        real = scalar.real
        imag = scalar.imag
        sign = "+" if imag >= 0 else "-"
        return f"{real:.6g}{sign}{abs(imag):.6g}j"
    if isinstance(scalar, int):
        return str(scalar)
    if isinstance(scalar, float):
        return f"{scalar:.6g}"
    return str(scalar)


def _format_term(blade_name: str, coefficient: Any) -> str:
    coeff = _normalize_scalar(coefficient)
    if blade_name == "e":
        return _format_scalar(coeff)
    if coeff == 1:
        return blade_name
    if coeff == -1:
        return f"-{blade_name}"
    return f"{_format_scalar(coeff)}*{blade_name}"


def _format_multivector(
    mv: MVArray,
    *,
    batch_shape: tuple[int, ...] | None = None,
    sample_index: tuple[int, ...] = (),
) -> str:
    if mv.layout.size == 0:
        return "0"

    target_batch_shape = mv.batch_shape if batch_shape is None else batch_shape
    values = _sample_values(mv, target_batch_shape, sample_index)

    parts: list[str] = []
    for blade, coefficient in zip(mv.layout.blades, values):
        if _is_zero(coefficient):
            continue
        parts.append(_format_term(mv.algebra.blade_name(blade), coefficient))

    if not parts:
        return "0"

    text = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            text += " - " + part[1:]
        else:
            text += " + " + part
    return text


def _resolve_sample_index(batch_shape: tuple[int, ...], sample: str | None) -> tuple[int, ...]:
    if not batch_shape:
        return ()
    if sample is None:
        return tuple(0 for _ in batch_shape)

    pieces = tuple(int(piece.strip()) for piece in sample.split(",") if piece.strip())
    if len(pieces) != len(batch_shape):
        shape_text = ", ".join(str(size) for size in batch_shape)
        raise ValueError(
            f"sample index {sample!r} does not match batch rank {len(batch_shape)} for shape ({shape_text})."
        )

    for index, size in zip(pieces, batch_shape):
        if index < 0 or index >= size:
            raise ValueError(f"sample index {pieces!r} is out of bounds for batch shape {batch_shape}.")
    return pieces


def _sample_values(
    mv: MVArray,
    batch_shape: tuple[int, ...],
    sample_index: tuple[int, ...],
) -> np.ndarray[Any, np.dtype[Any]]:
    values = np.broadcast_to(mv.values, batch_shape + (mv.layout.size,))
    return np.asarray(values[sample_index + (slice(None),)], dtype=mv.dtype)


def _build_trace(
    lhs: MVArray,
    rhs: MVArray,
    *,
    kind: OpKind,
    plan: OpPlan,
    result: MVArray,
    sample: str | None,
) -> ProductTrace:
    batch_shape = np.broadcast_shapes(lhs.batch_shape, rhs.batch_shape)
    sample_index = _resolve_sample_index(batch_shape, sample)

    lhs_values = _sample_values(lhs, batch_shape, sample_index)
    rhs_values = _sample_values(rhs, batch_shape, sample_index)
    result_values = _sample_values(result, batch_shape, sample_index)

    contributions: list[TermContribution] = []
    for term in plan.terms:
        lhs_value = _normalize_scalar(lhs_values[term.lhs_index])
        rhs_value = _normalize_scalar(rhs_values[term.rhs_index])
        value = _normalize_scalar(term.coefficient * lhs_value * rhs_value)
        contributions.append(
            TermContribution(
                lhs_blade=plan.lhs_blades[term.lhs_index],
                rhs_blade=plan.rhs_blades[term.rhs_index],
                out_blade=term.out_blade,
                coefficient=term.coefficient,
                lhs_value=lhs_value,
                rhs_value=rhs_value,
                value=value,
            )
        )

    return ProductTrace(
        kind=kind,
        operator_symbol=OPERATOR_SYMBOLS[kind],
        plan=plan,
        lhs_text=_format_multivector(lhs, batch_shape=batch_shape, sample_index=sample_index),
        rhs_text=_format_multivector(rhs, batch_shape=batch_shape, sample_index=sample_index),
        result_text=_format_multivector(result, batch_shape=batch_shape, sample_index=sample_index),
        batch_shape=batch_shape,
        sample_index=sample_index,
        lhs_values=tuple(_normalize_scalar(value) for value in lhs_values),
        rhs_values=tuple(_normalize_scalar(value) for value in rhs_values),
        result_values=tuple(_normalize_scalar(value) for value in result_values),
        contributions=tuple(contributions),
    )


@contextmanager
def _capture_binary_products(
    *,
    traces: list[ProductTrace],
    sample: str | None,
):
    original = ops_module._execute_binary_product

    def traced_execute(lhs: MVArray, rhs: MVArray, kind: OpKind) -> MVArray:
        result = original(lhs, rhs, kind)
        plan = ops_module.plan_binary_product(lhs.layout, rhs.layout, kind)
        traces.append(
            _build_trace(
                lhs,
                rhs,
                kind=kind,
                plan=plan,
                result=result,
                sample=sample,
            )
        )
        return result

    ops_module._execute_binary_product = traced_execute
    try:
        yield
    finally:
        ops_module._execute_binary_product = original


def _build_context(algebra_name: str) -> dict[str, Any]:
    algebra = Algebra.from_name(algebra_name)
    context: dict[str, Any] = {
        "alg": algebra,
        "Algebra": Algebra,
        "MVArray": MVArray,
        "np": np,
        "amsa": amsa,
    }
    for name in amsa.__all__:
        context[name] = getattr(amsa, name)
    return context


def trace_expression(
    *,
    algebra_name: str,
    statements: tuple[str, ...],
    expression: str,
    sample: str | None,
) -> TraceReport:
    context = _build_context(algebra_name)
    traces: list[ProductTrace] = []

    with _capture_binary_products(traces=traces, sample=sample):
        for statement in statements:
            exec(statement, context, context)
        result = eval(expression, context, context)

    if not isinstance(result, MVArray):
        raise TypeError(f"Expression must evaluate to an MVArray, got {type(result)!r}.")

    final_sample = _resolve_sample_index(result.batch_shape, sample)
    final_text = _format_multivector(result, batch_shape=result.batch_shape, sample_index=final_sample)
    return TraceReport(
        algebra_name=algebra_name,
        expression=expression,
        statements=statements,
        final_result_text=final_text,
        traces=tuple(traces),
    )


def _node_positions(items: tuple[str, ...], *, top: int, step: int) -> dict[str, int]:
    return {item: top + index * step for index, item in enumerate(items)}


def _trace_output_totals(trace: ProductTrace) -> list[tuple[str, str, str]]:
    totals: list[tuple[str, str, str]] = []
    for blade, value in zip(trace.plan.output_blades, trace.result_values):
        name = trace.plan.algebra.blade_name(blade)
        if _is_zero(value):
            state = "cancelled" if any(
                contribution.out_blade == blade and not _is_zero(contribution.value)
                for contribution in trace.contributions
            ) else "inactive"
        else:
            state = "active"
        totals.append((name, _format_scalar(value), state))
    return totals


def _trace_board(trace: ProductTrace, step_index: int) -> str:
    lhs_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.lhs_blades)
    rhs_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.rhs_blades)
    out_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.output_blades)
    term_step = 92
    blade_step = 88
    height = max(360, 140 + term_step * max(len(trace.contributions), 1))

    lhs_x = 48
    rhs_x = 230
    term_x = 470
    out_x = 820

    lhs_pos = _node_positions(lhs_names, top=72, step=blade_step)
    rhs_pos = _node_positions(rhs_names, top=72, step=blade_step)
    out_pos = _node_positions(out_names, top=72, step=blade_step)

    active_lhs_names = {
        trace.plan.algebra.blade_name(blade)
        for blade, value in zip(trace.plan.lhs_blades, trace.lhs_values)
        if not _is_zero(value)
    }
    active_rhs_names = {
        trace.plan.algebra.blade_name(blade)
        for blade, value in zip(trace.plan.rhs_blades, trace.rhs_values)
        if not _is_zero(value)
    }
    active_terms = [contribution for contribution in trace.contributions if not _is_zero(contribution.value)]
    active_output_names = {
        trace.plan.algebra.blade_name(contribution.out_blade)
        for contribution in active_terms
    }
    result_output_names = {
        trace.plan.algebra.blade_name(blade)
        for blade, value in zip(trace.plan.output_blades, trace.result_values)
        if not _is_zero(value)
    }
    output_totals = {name: value for name, value, _ in _trace_output_totals(trace)}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for name, y in lhs_pos.items():
        state = "active" if name in active_lhs_names else "inactive"
        nodes.append(
            {
                "id": f"step{step_index}-lhs-{name}",
                "x": lhs_x,
                "y": y,
                "kind": "blade",
                "state": state,
                "title": name,
                "subtitle": "lhs",
            }
        )

    for name, y in rhs_pos.items():
        state = "active" if name in active_rhs_names else "inactive"
        nodes.append(
            {
                "id": f"step{step_index}-rhs-{name}",
                "x": rhs_x,
                "y": y,
                "kind": "blade",
                "state": state,
                "title": name,
                "subtitle": "rhs",
            }
        )

    for index, contribution in enumerate(trace.contributions):
        term_y = 72 + index * term_step
        lhs_name = trace.plan.algebra.blade_name(contribution.lhs_blade)
        rhs_name = trace.plan.algebra.blade_name(contribution.rhs_blade)
        out_name = trace.plan.algebra.blade_name(contribution.out_blade)
        active = not _is_zero(contribution.value)
        polarity = (
            "positive"
            if np.sign(np.real(np.asarray(contribution.value))).item() >= 0
            else "negative"
        )
        state = polarity if active else "inactive"
        term_id = f"step{step_index}-term-{index}"
        nodes.append(
            {
                "id": term_id,
                "x": term_x,
                "y": term_y,
                "kind": "term",
                "state": state,
                "title": f"{lhs_name} x {rhs_name}",
                "subtitle": (
                    f"{contribution.coefficient} * {_format_scalar(contribution.lhs_value)}"
                    f" * {_format_scalar(contribution.rhs_value)} = {_format_scalar(contribution.value)}"
                ),
            }
        )
        edges.extend(
            [
                {
                    "from": f"step{step_index}-lhs-{lhs_name}",
                    "to": term_id,
                    "state": state,
                },
                {
                    "from": f"step{step_index}-rhs-{rhs_name}",
                    "to": term_id,
                    "state": state,
                },
                {
                    "from": term_id,
                    "to": f"step{step_index}-out-{out_name}",
                    "state": state,
                },
            ]
        )

    for name, y in out_pos.items():
        if name in result_output_names:
            state = "active"
        elif name in active_output_names:
            state = "cancelled"
        else:
            state = "inactive"
        nodes.append(
            {
                "id": f"step{step_index}-out-{name}",
                "x": out_x,
                "y": y,
                "kind": "output",
                "state": state,
                "title": name,
                "subtitle": f"total = {output_totals[name]}",
            }
        )

    totals_markup = "".join(
        (
            f'<div class="total-chip {state}">'
            f'<span class="total-name">{html.escape(name)}</span>'
            f'<span class="total-value">{html.escape(value)}</span>'
            "</div>"
        )
        for name, value, state in _trace_output_totals(trace)
    )
    nodes_markup = "".join(
        (
            f'<div class="node {node["kind"]} {node["state"]}" '
            f'data-node-id="{html.escape(str(node["id"]))}" '
            f'style="left:{int(node["x"])}px; top:{int(node["y"])}px;">'
            f'<div class="node-title">{html.escape(str(node["title"]))}</div>'
            f'<div class="node-subtitle">{html.escape(str(node["subtitle"]))}</div>'
            "</div>"
        )
        for node in nodes
    )
    edges_json = html.escape(json.dumps(edges), quote=True)
    return (
        '<div class="board-wrap">'
        '<div class="board-head">'
        '<div class="column-kicker">lhs</div>'
        '<div class="column-kicker">rhs</div>'
        '<div class="column-kicker">terms</div>'
        '<div class="column-kicker">outputs</div>'
        "</div>"
        f'<div class="totals-row">{totals_markup}</div>'
        f'<div class="graph-board" data-edges="{edges_json}" style="height:{height}px;">'
        '<svg class="graph-wires" aria-hidden="true"></svg>'
        f"{nodes_markup}"
        "</div>"
        '<p class="drag-note">drag any node to rearrange the board; wires update live.</p>'
        "</div>"
    )


def render_report(report: TraceReport) -> str:
    statement_lines = "\n".join(report.statements) if report.statements else "# no setup statements"
    sample_text = "scalar sample"
    if report.traces:
        first_sample = report.traces[0].sample_index
        sample_text = str(first_sample) if first_sample else "scalar sample"

    sections: list[str] = []
    for index, trace in enumerate(report.traces):
        structural_terms = len(trace.contributions)
        active_terms = sum(not _is_zero(contribution.value) for contribution in trace.contributions)
        sections.append(
            "<section class=\"step-card\">"
            f"<div class=\"step-meta\">step {index + 1}: {html.escape(trace.kind)} ({html.escape(trace.operator_symbol)})</div>"
            f"<h2>{html.escape(trace.lhs_text)} {html.escape(trace.operator_symbol)} {html.escape(trace.rhs_text)}</h2>"
            f"<p class=\"result-text\">result: {html.escape(trace.result_text)}</p>"
            f"<p class=\"step-summary\">structural terms: {structural_terms} | active terms: {active_terms}</p>"
            f"{_trace_board(trace, index)}"
            "</section>"
        )

    if not sections:
        sections.append(
            "<section class=\"step-card\">"
            "<div class=\"step-meta\">no binary product steps captured</div>"
            "<h2>This expression did not trigger AMSA's binary product planner.</h2>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>amsa.lab</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #111315;
  --panel: rgba(22, 24, 27, 0.94);
  --panel-edge: rgba(255, 255, 255, 0.08);
  --text: #f1efe8;
  --muted: #aaa59a;
  --accent: #d8c3a5;
  --line: rgba(255, 255, 255, 0.12);
  --positive: #9fcf8f;
  --negative: #cf8f8f;
  --inactive: #4f5358;
  --cancelled: #d0b276;
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  font-family: "IBM Plex Mono", "Iosevka", "SFMono-Regular", "Consolas", monospace;
  background:
    radial-gradient(circle at top left, rgba(216, 195, 165, 0.10), transparent 30%),
    linear-gradient(180deg, #17191c 0%, #101214 100%);
  color: var(--text);
}}
main {{
  width: min(1180px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 36px 0 56px;
}}
.hero,
.step-card {{
  background: var(--panel);
  border: 1px solid var(--panel-edge);
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
}}
.hero {{
  padding: 22px;
  margin-bottom: 24px;
}}
.eyebrow {{
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  margin-bottom: 10px;
}}
h1,
h2 {{
  margin: 0;
  font-weight: 600;
}}
h1 {{
  font-size: clamp(28px, 5vw, 46px);
  line-height: 1.05;
}}
h2 {{
  font-size: 21px;
  line-height: 1.35;
}}
.hero-grid {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin-top: 18px;
}}
.stat {{
  padding: 14px 16px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
}}
.stat-label {{
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}}
.stat-value {{
  font-size: 15px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}}
.step-card {{
  padding: 22px;
  margin-bottom: 22px;
}}
.step-meta,
.step-summary,
.result-text {{
  color: var(--muted);
}}
.step-meta {{
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 10px;
}}
.result-text {{
  margin: 12px 0 8px;
}}
.step-summary {{
  margin: 0 0 16px;
}}
.board-wrap {{
  margin-top: 18px;
}}
.board-head {{
  display: grid;
  grid-template-columns: 1fr 1fr 1.25fr 1fr;
  gap: 8px;
  margin-bottom: 10px;
}}
.column-kicker {{
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 11px;
}}
.totals-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}}
.total-chip {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.03);
  font-size: 12px;
}}
.total-chip.active {{
  border-color: rgba(159, 207, 143, 0.35);
}}
.total-chip.cancelled {{
  border-color: rgba(208, 178, 118, 0.4);
}}
.total-name {{
  color: var(--muted);
}}
.total-value {{
  color: var(--text);
}}
.graph-board {{
  position: relative;
  overflow: hidden;
  min-height: 360px;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(180deg, #17191b, #121416);
  background-size: 24px 24px, 24px 24px, auto;
}}
.graph-wires {{
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}}
.wire {{
  fill: none;
  stroke-width: 2;
}}
.wire.inactive {{
  stroke: rgba(255, 255, 255, 0.10);
}}
.wire.positive {{
  stroke: rgba(159, 207, 143, 0.70);
}}
.wire.negative {{
  stroke: rgba(207, 143, 143, 0.70);
}}
.node {{
  position: absolute;
  width: 132px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(27, 29, 33, 0.96);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
  cursor: grab;
  user-select: none;
  touch-action: none;
}}
.node.blade {{
  width: 110px;
}}
.node.output {{
  width: 136px;
}}
.node.term {{
  width: 182px;
}}
.node.active {{
  border-color: rgba(216, 195, 165, 0.24);
}}
.node.positive {{
  border-color: rgba(159, 207, 143, 0.30);
}}
.node.negative {{
  border-color: rgba(207, 143, 143, 0.30);
}}
.node.cancelled {{
  border-color: rgba(208, 178, 118, 0.34);
}}
.node.inactive {{
  opacity: 0.72;
}}
.node.dragging {{
  cursor: grabbing;
  z-index: 4;
}}
.node-title {{
  font-size: 14px;
  line-height: 1.2;
  color: var(--text);
}}
.node-subtitle {{
  margin-top: 6px;
  font-size: 11px;
  line-height: 1.4;
  color: var(--muted);
}}
.drag-note {{
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 12px;
}}
@media (max-width: 720px) {{
  main {{
    width: min(100vw - 22px, 1180px);
    padding-top: 14px;
  }}
  .hero,
  .step-card {{
    border-radius: 14px;
    padding: 18px;
  }}
  .board-head {{
    grid-template-columns: repeat(2, 1fr);
  }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div class="eyebrow">amsa.lab probe</div>
    <h1>{html.escape(report.expression)}</h1>
    <div class="hero-grid">
      <div class="stat">
        <div class="stat-label">algebra</div>
        <div class="stat-value">{html.escape(report.algebra_name)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">final result</div>
        <div class="stat-value">{html.escape(report.final_result_text)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">setup</div>
        <div class="stat-value">{html.escape(statement_lines)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">sample</div>
        <div class="stat-value">{html.escape(sample_text)}</div>
      </div>
    </div>
  </section>
  {"".join(sections)}
</main>
<script>
const boards = document.querySelectorAll(".graph-board");

function boardPoint(board, node) {{
  return {{
    x: node.offsetLeft + node.offsetWidth / 2,
    y: node.offsetTop + node.offsetHeight / 2,
  }};
}}

function curvePath(fromPoint, toPoint) {{
  const dx = (toPoint.x - fromPoint.x) * 0.45;
  return `M ${{fromPoint.x}} ${{fromPoint.y}} C ${{fromPoint.x + dx}} ${{fromPoint.y}}, ${{toPoint.x - dx}} ${{toPoint.y}}, ${{toPoint.x}} ${{toPoint.y}}`;
}}

function renderBoard(board) {{
  const svg = board.querySelector(".graph-wires");
  const edges = JSON.parse(board.dataset.edges);
  const nodes = new Map(
    Array.from(board.querySelectorAll(".node")).map((node) => [node.dataset.nodeId, node])
  );
  svg.innerHTML = "";
  for (const edge of edges) {{
    const fromNode = nodes.get(edge.from);
    const toNode = nodes.get(edge.to);
    if (!fromNode || !toNode) {{
      continue;
    }}
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", `wire ${{edge.state}}`);
    path.setAttribute("d", curvePath(boardPoint(board, fromNode), boardPoint(board, toNode)));
    svg.appendChild(path);
  }}
}}

for (const board of boards) {{
  let drag = null;

  renderBoard(board);

  board.addEventListener("pointerdown", (event) => {{
    const node = event.target.closest(".node");
    if (!node || !board.contains(node)) {{
      return;
    }}
    const rect = board.getBoundingClientRect();
    drag = {{
      node,
      offsetX: event.clientX - rect.left - node.offsetLeft,
      offsetY: event.clientY - rect.top - node.offsetTop,
    }};
    node.classList.add("dragging");
    node.setPointerCapture(event.pointerId);
  }});

  board.addEventListener("pointermove", (event) => {{
    if (!drag) {{
      return;
    }}
    const rect = board.getBoundingClientRect();
    const nextLeft = Math.max(0, Math.min(rect.width - drag.node.offsetWidth, event.clientX - rect.left - drag.offsetX));
    const nextTop = Math.max(0, Math.min(rect.height - drag.node.offsetHeight, event.clientY - rect.top - drag.offsetY));
    drag.node.style.left = `${{nextLeft}}px`;
    drag.node.style.top = `${{nextTop}}px`;
    renderBoard(board);
  }});

  function endDrag(event) {{
    if (!drag) {{
      return;
    }}
    drag.node.classList.remove("dragging");
    if (event && drag.node.hasPointerCapture(event.pointerId)) {{
      drag.node.releasePointerCapture(event.pointerId);
    }}
    drag = null;
  }}

  board.addEventListener("pointerup", endDrag);
  board.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", () => renderBoard(board));
}}
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal external AMSA visual debugger probe. "
            "This is a trusted local runner that traces binary AMSA products "
            "and renders a self-contained HTML report."
        )
    )
    parser.add_argument(
        "--algebra",
        default=DEFAULT_ALGEBRA,
        help=f"Algebra preset passed to Algebra.from_name(...). Default: {DEFAULT_ALGEBRA}.",
    )
    parser.add_argument(
        "--stmt",
        action="append",
        default=None,
        help="Setup statement executed before the final expression. May be repeated.",
    )
    parser.add_argument(
        "--expr",
        default=DEFAULT_EXPRESSION,
        help=f"Expression to evaluate. Default: {DEFAULT_EXPRESSION!r}.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Optional comma-separated batch index, such as '0' or '0,1'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML output path. Default: {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statements = tuple(args.stmt) if args.stmt is not None else DEFAULT_STATEMENTS
    report = trace_expression(
        algebra_name=args.algebra,
        statements=statements,
        expression=args.expr,
        sample=args.sample,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report), encoding="utf-8")
    print(f"wrote amsa.lab report to {args.output}")
    print(f"expression: {args.expr}")
    print(f"final result: {report.final_result_text}")
    print(f"captured binary steps: {len(report.traces)}")


if __name__ == "__main__":
    main()
