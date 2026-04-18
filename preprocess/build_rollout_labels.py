from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocess.common import read_jsonl, write_jsonl
from utils.diagnostics import ensure_dir, print_event, save_bar_png, write_json


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
GO_UP = 4
GO_DOWN = 5
MOVE_LEFT = 6
MOVE_RIGHT = 7

OPPOSITES = {
    TURN_LEFT: TURN_RIGHT,
    TURN_RIGHT: TURN_LEFT,
    GO_UP: GO_DOWN,
    GO_DOWN: GO_UP,
    MOVE_LEFT: MOVE_RIGHT,
    MOVE_RIGHT: MOVE_LEFT,
}


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    rows = []
    windows = list(read_jsonl(args.step_windows))
    action_positive = {str(i): 0.0 for i in range(8)}
    action_cost = {str(i): 0.0 for i in range(8)}
    print_event("build_rollout_labels", "start", windows=len(windows))
    for row in windows:
        gt_action = int(row["gt_action"])
        next_action = int(row.get("next_action", gt_action))
        completion = float(row.get("completion", row.get("global_completion", 0.0)))
        global_completion = float(row.get("global_completion", completion))
        next_global_completion = float(row.get("next_global_completion", global_completion))
        milestone_changes = int(row.get("next_milestone_id", row.get("milestone_id", 1))) != int(row.get("milestone_id", 1))
        recent_actions = [int(action) for action in (row.get("action_history") or []) if int(action) != STOP]
        labels: Dict[str, Dict[str, float]] = {}
        for action_id in row.get("legal_action_ids", list(range(8))):
            action_id = int(action_id)
            progress = _progress_label(
                action_id=action_id,
                gt_action=gt_action,
                next_action=next_action,
                completion=completion,
                global_delta=max(0.0, next_global_completion - global_completion),
                milestone_changes=milestone_changes,
            )
            cost = _cost_label(
                action_id=action_id,
                gt_action=gt_action,
                next_action=next_action,
                completion=completion,
                recent_actions=recent_actions,
                milestone_changes=milestone_changes,
            )
            labels[str(action_id)] = {"progress": progress, "cost": cost}
            action_positive[str(action_id)] += progress
            action_cost[str(action_id)] += cost
        rows.append({"sample_id": row["sample_id"], "labels": labels})
    write_jsonl(args.output, rows)
    denom = max(1, len(windows))
    save_bar_png(diag_dir / "positive_progress_by_action.png", "Positive progress labels by action", action_positive)
    save_bar_png(diag_dir / "average_cost_by_action.png", "Average cost label by action", {k: v / denom for k, v in action_cost.items()})
    write_json(diag_dir / "summary.json", {"step_windows": len(windows), "label_rows": len(rows)})
    print_event("build_rollout_labels", "done", label_rows=len(rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--output", default="data/rollout_labels/train.jsonl")
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_rollout_labels")
    build(parser.parse_args())


def _progress_label(
    action_id: int,
    gt_action: int,
    next_action: int,
    completion: float,
    global_delta: float,
    milestone_changes: bool,
) -> float:
    if action_id == gt_action:
        return float(np.clip(0.65 + 2.0 * global_delta + (0.2 if milestone_changes else 0.0), 0.0, 1.0))
    if action_id == next_action:
        return 0.45
    if action_id == MOVE_FORWARD and gt_action in {TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT}:
        return 0.2
    if action_id == STOP:
        return 0.35 if completion >= 0.9 else 0.0
    return 0.05


def _cost_label(
    action_id: int,
    gt_action: int,
    next_action: int,
    completion: float,
    recent_actions: Iterable[int],
    milestone_changes: bool,
) -> float:
    if action_id == gt_action:
        return 0.0

    cost = 0.75
    if action_id == next_action:
        cost = min(cost, 0.35)
    if action_id == STOP:
        cost = min(cost, 0.2) if completion >= 0.9 or milestone_changes else 1.0
    if OPPOSITES.get(gt_action) == action_id:
        cost = max(cost, 0.9)
    if _is_repeated_ineffective(action_id, recent_actions):
        cost = max(cost, 0.85)
    if action_id == MOVE_FORWARD and gt_action in {TURN_LEFT, TURN_RIGHT}:
        cost = max(cost, 0.65)
    return float(np.clip(cost, 0.0, 1.0))


def _is_repeated_ineffective(action_id: int, recent_actions: Iterable[int]) -> bool:
    recent = list(recent_actions)[-3:]
    return len(recent) >= 2 and all(action == action_id for action in recent[-2:]) and action_id != MOVE_FORWARD


if __name__ == "__main__":
    main()
