from __future__ import annotations

import argparse
import html
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
DEFAULT_OUTPUT = Path("/tmp/amsa_lab.html")

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


def _curve_path(x0: int, y0: float, x1: int, y1: float) -> str:
    control_x = (x0 + x1) / 2
    return f"M {x0} {y0:.1f} C {control_x:.1f} {y0:.1f}, {control_x:.1f} {y1:.1f}, {x1} {y1:.1f}"


def _node_positions(items: tuple[str, ...], *, top: int, step: int) -> dict[str, float]:
    return {item: float(top + index * step) for index, item in enumerate(items)}


def _trace_svg(trace: ProductTrace, step_index: int) -> str:
    lhs_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.lhs_blades)
    rhs_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.rhs_blades)
    out_names = tuple(trace.plan.algebra.blade_name(blade) for blade in trace.plan.output_blades)
    term_step = 72
    blade_step = 84
    height = max(
        300,
        120 + term_step * max(len(trace.contributions), 1),
        120 + blade_step * max(len(lhs_names), len(rhs_names), len(out_names), 1),
    )

    lhs_x = 110
    rhs_x = 300
    term_x = 555
    out_x = 835

    lhs_pos = _node_positions(lhs_names, top=90, step=blade_step)
    rhs_pos = _node_positions(rhs_names, top=90, step=blade_step)
    out_pos = _node_positions(out_names, top=90, step=blade_step)

    active_terms = [contribution for contribution in trace.contributions if not _is_zero(contribution.value)]
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
    active_output_names = {
        trace.plan.algebra.blade_name(contribution.out_blade)
        for contribution in active_terms
    }
    result_output_names = {
        trace.plan.algebra.blade_name(blade)
        for blade, value in zip(trace.plan.output_blades, trace.result_values)
        if not _is_zero(value)
    }

    svg_parts = [
        f'<svg viewBox="0 0 960 {height}" role="img" aria-label="Blade flow graph for step {step_index + 1}">',
        '<text class="column-title" x="110" y="46">lhs blades</text>',
        '<text class="column-title" x="300" y="46">rhs blades</text>',
        '<text class="column-title" x="555" y="46">plan terms</text>',
        '<text class="column-title" x="835" y="46">output blades</text>',
    ]

    for index, contribution in enumerate(trace.contributions):
        term_y = 90 + index * term_step
        lhs_name = trace.plan.algebra.blade_name(contribution.lhs_blade)
        rhs_name = trace.plan.algebra.blade_name(contribution.rhs_blade)
        out_name = trace.plan.algebra.blade_name(contribution.out_blade)
        active = not _is_zero(contribution.value)
        polarity = "positive" if np.sign(np.real(np.asarray(contribution.value))).item() >= 0 else "negative"
        classes = "flow active" if active else "flow inactive"
        if active:
            classes += f" {polarity}"

        delay = f"{index * 0.18:.2f}s"
        svg_parts.append(
            f'<path class="{classes}" style="animation-delay:{delay}" d="{_curve_path(lhs_x + 58, lhs_pos[lhs_name], term_x - 72, term_y)}" />'
        )
        svg_parts.append(
            f'<path class="{classes}" style="animation-delay:{delay}" d="{_curve_path(rhs_x + 58, rhs_pos[rhs_name], term_x - 72, term_y)}" />'
        )
        svg_parts.append(
            f'<path class="{classes}" style="animation-delay:{delay}" d="{_curve_path(term_x + 72, term_y, out_x - 58, out_pos[out_name])}" />'
        )

        term_classes = "term-node active" if active else "term-node inactive"
        if active:
            term_classes += f" {polarity}"
        svg_parts.append(
            f'<g class="{term_classes}" style="animation-delay:{delay}">'
            f'<rect x="{term_x - 72}" y="{term_y - 24}" width="144" height="48" rx="14" />'
            f'<text class="term-label" x="{term_x}" y="{term_y - 4}">{html.escape(lhs_name)} x {html.escape(rhs_name)}</text>'
            f'<text class="term-detail" x="{term_x}" y="{term_y + 13}">c={contribution.coefficient}, value={html.escape(_format_scalar(contribution.value))}</text>'
            "</g>"
        )

    for name, y in lhs_pos.items():
        node_class = "blade-node active" if name in active_lhs_names else "blade-node inactive"
        svg_parts.append(
            f'<g class="{node_class}"><circle cx="{lhs_x}" cy="{y}" r="28" /><text x="{lhs_x}" y="{y + 5}">{html.escape(name)}</text></g>'
        )

    for name, y in rhs_pos.items():
        node_class = "blade-node active" if name in active_rhs_names else "blade-node inactive"
        svg_parts.append(
            f'<g class="{node_class}"><circle cx="{rhs_x}" cy="{y}" r="28" /><text x="{rhs_x}" y="{y + 5}">{html.escape(name)}</text></g>'
        )

    for name, y in out_pos.items():
        if name in result_output_names:
            output_class = "blade-node active"
        elif name in active_output_names:
            output_class = "blade-node cancelled"
        else:
            output_class = "blade-node inactive"
        svg_parts.append(
            f'<g class="{output_class}"><circle cx="{out_x}" cy="{y}" r="28" /><text x="{out_x}" y="{y + 5}">{html.escape(name)}</text></g>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


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
            f"{_trace_svg(trace, index)}"
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
  --bg: #06141f;
  --panel: rgba(8, 28, 43, 0.84);
  --panel-edge: rgba(132, 196, 255, 0.18);
  --text: #e9f4ff;
  --muted: #95afc6;
  --accent: #79d2ff;
  --positive: #6cf1a7;
  --negative: #ff7c7c;
  --inactive: rgba(151, 180, 205, 0.22);
  --cancelled: #ffd36b;
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  font-family: "Iosevka", "SFMono-Regular", "Consolas", monospace;
  background:
    radial-gradient(circle at top left, rgba(121, 210, 255, 0.16), transparent 32%),
    radial-gradient(circle at top right, rgba(108, 241, 167, 0.10), transparent 28%),
    linear-gradient(180deg, #041018 0%, #081a27 44%, #030c13 100%);
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
  border-radius: 24px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.30);
  backdrop-filter: blur(12px);
}}
.hero {{
  padding: 28px;
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
  gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin-top: 22px;
}}
.stat {{
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(8, 33, 49, 0.70);
  border: 1px solid rgba(132, 196, 255, 0.12);
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
svg {{
  width: 100%;
  height: auto;
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(5, 21, 33, 0.96), rgba(4, 17, 26, 0.90)),
    linear-gradient(90deg, rgba(121, 210, 255, 0.06), transparent 20%, transparent 80%, rgba(108, 241, 167, 0.06));
  border: 1px solid rgba(132, 196, 255, 0.10);
}}
.column-title {{
  fill: var(--muted);
  font-size: 12px;
  text-anchor: middle;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}}
.flow {{
  fill: none;
  stroke-width: 2.6;
}}
.flow.inactive {{
  stroke: var(--inactive);
}}
.flow.active.positive {{
  stroke: var(--positive);
  stroke-dasharray: 10 8;
  animation: dash 2.4s linear infinite;
}}
.flow.active.negative {{
  stroke: var(--negative);
  stroke-dasharray: 10 8;
  animation: dash 2.4s linear infinite;
}}
.blade-node circle {{
  stroke-width: 2;
}}
.blade-node text,
.term-node text {{
  fill: var(--text);
  text-anchor: middle;
}}
.blade-node.active circle {{
  fill: rgba(16, 63, 95, 0.95);
  stroke: rgba(121, 210, 255, 0.72);
}}
.blade-node.inactive circle {{
  fill: rgba(20, 35, 48, 0.82);
  stroke: rgba(151, 180, 205, 0.18);
}}
.blade-node.cancelled circle {{
  fill: rgba(61, 48, 13, 0.88);
  stroke: rgba(255, 211, 107, 0.62);
}}
.blade-node text {{
  font-size: 15px;
  font-weight: 600;
}}
.term-node rect {{
  stroke-width: 1.8;
}}
.term-node.active rect {{
  fill: rgba(13, 40, 59, 0.96);
  stroke: rgba(121, 210, 255, 0.56);
  animation: pulse 2.0s ease-in-out infinite;
}}
.term-node.active.positive rect {{
  box-shadow: 0 0 10px rgba(108, 241, 167, 0.35);
}}
.term-node.active.negative rect {{
  box-shadow: 0 0 10px rgba(255, 124, 124, 0.35);
}}
.term-node.inactive rect {{
  fill: rgba(17, 31, 43, 0.80);
  stroke: rgba(151, 180, 205, 0.16);
}}
.term-label {{
  font-size: 14px;
  font-weight: 600;
}}
.term-detail {{
  font-size: 11px;
  fill: var(--muted);
}}
@keyframes dash {{
  to {{
    stroke-dashoffset: -36;
  }}
}}
@keyframes pulse {{
  0%,
  100% {{
    transform: translateY(0px);
  }}
  50% {{
    transform: translateY(-1px);
  }}
}}
@media (max-width: 720px) {{
  main {{
    width: min(100vw - 22px, 1180px);
    padding-top: 14px;
  }}
  .hero,
  .step-card {{
    border-radius: 18px;
    padding: 18px;
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
