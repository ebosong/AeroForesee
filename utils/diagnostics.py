from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_event(stage: str, message: str, **fields: Any) -> None:
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" | {detail}" if detail else ""
    print(f"[{stage}] {message}{suffix}", flush=True)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, ensure_ascii=False, indent=2)


def save_bar_svg(
    path: str | Path,
    title: str,
    values: Mapping[str, float],
    width: int = 840,
    bar_height: int = 26,
    save_png: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(values.items())
    max_value = max([abs(float(v)) for _, v in items] + [1.0])
    height = 70 + max(1, len(items)) * (bar_height + 12)
    rows: List[str] = []
    for idx, (label, value) in enumerate(items):
        y = 52 + idx * (bar_height + 12)
        bar_width = int((width - 260) * abs(float(value)) / max_value)
        rows.append(f'<text x="24" y="{y + 18}" font-size="14" fill="#1f2937">{_esc(label)}</text>')
        rows.append(f'<rect x="210" y="{y}" width="{bar_width}" height="{bar_height}" rx="4" fill="#2563eb" />')
        rows.append(f'<text x="{220 + bar_width}" y="{y + 18}" font-size="13" fill="#111827">{float(value):.4f}</text>')
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="24" y="30" font-size="20" font-family="Arial" font-weight="700" fill="#111827">{_esc(title)}</text>',
        *rows,
        "</svg>",
    ])
    path.write_text(svg, encoding="utf-8")
    if save_png:
        save_bar_png(path.with_suffix(".png"), title, values, width=width)


def save_line_svg(
    path: str | Path,
    title: str,
    values: Iterable[float],
    width: int = 840,
    height: int = 360,
    save_png: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [float(v) for v in values]
    if not vals:
        vals = [0.0]
    v_min = min(vals)
    v_max = max(vals)
    if math.isclose(v_min, v_max):
        v_max = v_min + 1.0
    left, top, right, bottom = 56, 48, width - 24, height - 44
    points = []
    for idx, value in enumerate(vals):
        x = left if len(vals) == 1 else left + (right - left) * idx / (len(vals) - 1)
        y = bottom - (bottom - top) * (value - v_min) / (v_max - v_min)
        points.append(f"{x:.1f},{y:.1f}")
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="24" y="30" font-size="20" font-family="Arial" font-weight="700" fill="#111827">{_esc(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af" />',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#dc2626" stroke-width="3" />',
        f'<text x="24" y="{top + 4}" font-size="12" fill="#374151">max {v_max:.4f}</text>',
        f'<text x="24" y="{bottom}" font-size="12" fill="#374151">min {v_min:.4f}</text>',
        "</svg>",
    ])
    path.write_text(svg, encoding="utf-8")
    if save_png:
        save_line_png(path.with_suffix(".png"), title, vals, width=width, height=height)


def save_bar_png(
    path: str | Path,
    title: str,
    values: Mapping[str, float],
    width: int = 840,
    height: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(values.items())
    if not items:
        items = [("empty", 0.0)]
    labels = [str(k) for k, _ in items]
    vals = [float(v) for _, v in items]
    height = height or 70 + max(1, len(items)) * 38
    fig, ax = _make_figure(width, height)
    y_positions = list(range(len(items)))
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in vals]
    ax.barh(y_positions, vals, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    span = max(abs(v) for v in vals) or 1.0
    ax.set_xlim(min(0.0, min(vals)) - 0.12 * span, max(0.0, max(vals)) + 0.18 * span)
    for y, value in zip(y_positions, vals):
        offset = 0.02 * span if value >= 0 else -0.02 * span
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, y, f"{value:.4f}", va="center", ha=ha, fontsize=8, color="#111827")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    _close_figure(fig)


def save_line_png(
    path: str | Path,
    title: str,
    values: Iterable[float],
    width: int = 840,
    height: int = 360,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [float(v) for v in values]
    if not vals:
        vals = [0.0]
    fig, ax = _make_figure(width, height)
    ax.plot(list(range(len(vals))), vals, color="#dc2626", linewidth=2.5, marker="o", markersize=4)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    _close_figure(fig)


def _esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_figure(width: int, height: int):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)


def _close_figure(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)

