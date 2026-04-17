from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


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


def _make_figure(width: int, height: int):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)


def _close_figure(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)

