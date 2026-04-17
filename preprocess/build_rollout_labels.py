from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocess.common import read_jsonl, write_jsonl
from utils.diagnostics import ensure_dir, print_event, save_bar_svg, write_json


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    rows = []
    windows = list(read_jsonl(args.step_windows))
    action_positive = {str(i): 0.0 for i in range(8)}
    action_cost = {str(i): 0.0 for i in range(8)}
    print_event("build_rollout_labels", "start", windows=len(windows))
    for row in windows:
        gt_action = int(row["gt_action"])
        labels: Dict[str, Dict[str, float]] = {}
        for action_id in row.get("legal_action_ids", list(range(8))):
            action_id = int(action_id)
            progress = 1.0 if action_id == gt_action else 0.0
            cost = 0.0 if action_id == gt_action else 1.0
            if action_id == int(row.get("next_action", gt_action)):
                cost = min(cost, 0.5)
            labels[str(action_id)] = {"progress": progress, "cost": cost}
            action_positive[str(action_id)] += progress
            action_cost[str(action_id)] += cost
        rows.append({"sample_id": row["sample_id"], "labels": labels})
    write_jsonl(args.output, rows)
    denom = max(1, len(windows))
    save_bar_svg(diag_dir / "positive_progress_by_action.svg", "Positive progress labels by action", action_positive)
    save_bar_svg(diag_dir / "average_cost_by_action.svg", "Average cost label by action", {k: v / denom for k, v in action_cost.items()})
    write_json(diag_dir / "summary.json", {"step_windows": len(windows), "label_rows": len(rows)})
    print_event("build_rollout_labels", "done", label_rows=len(rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--output", default="data/rollout_labels/train.jsonl")
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_rollout_labels")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
