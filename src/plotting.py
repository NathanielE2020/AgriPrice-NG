"""Dependency-free SVG plots for actual-versus-predicted test values."""

from __future__ import annotations

import html
from pathlib import Path


def _points(values, x0, y0, width, height, low, high):
    span = high - low if high != low else 1.0
    result = []
    n = max(len(values) - 1, 1)
    for index, value in enumerate(values):
        x = x0 + width * index / n
        y = y0 + height - (float(value) - low) / span * height
        result.append(f"{x:.1f},{y:.1f}")
    return " ".join(result)


def write_actual_predicted_svg(
    path: str | Path,
    title: str,
    months: list[str],
    actual: list[float],
    predicted: list[float],
    baseline: list[float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(actual) + list(predicted) + (list(baseline) if baseline else [])
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    padding = (high - low) * 0.08 if high != low else 1.0
    low -= padding
    high += padding
    width, height = 1000, 560
    x0, y0, plot_w, plot_h = 80, 70, 860, 400
    actual_points = _points(actual, x0, y0, plot_w, plot_h, low, high)
    predicted_points = _points(predicted, x0, y0, plot_w, plot_h, low, high)
    baseline_points = (
        _points(baseline, x0, y0, plot_w, plot_h, low, high) if baseline else ""
    )
    month_labels = ""
    if months:
        for index in sorted({0, len(months) // 2, len(months) - 1}):
            x = x0 + plot_w * index / max(len(months) - 1, 1)
            month_labels += (
                f'<text x="{x:.1f}" y="500" text-anchor="middle" '
                f'font-size="12">{html.escape(months[index])}</text>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="500" y="30" text-anchor="middle" font-size="18" font-family="Arial">{html.escape(title)}</text>
<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#444"/>
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#444"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{actual_points}"/>
<polyline fill="none" stroke="#d62728" stroke-width="2" points="{predicted_points}"/>
{f'<polyline fill="none" stroke="#777" stroke-width="2" stroke-dasharray="6 4" points="{baseline_points}"/>' if baseline else ''}
<text x="{x0 + 10}" y="{y0 + 20}" font-size="13" fill="#1f77b4">Actual</text>
<text x="{x0 + 80}" y="{y0 + 20}" font-size="13" fill="#d62728">Predicted</text>
{f'<text x="{x0 + 180}" y="{y0 + 20}" font-size="13" fill="#777">Naive last-calendar-month</text>' if baseline else ''}
<text x="25" y="{y0 + plot_h / 2}" transform="rotate(-90 25 {y0 + plot_h / 2})" font-size="13">Price</text>
{month_labels}
</svg>'''
    path.write_text(svg, encoding="utf-8")
