from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
    geometric_labels = 0
    heuristic_labels = 0
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
            geometric = _geometric_consequence_label(row, action_id, completion, milestone_changes)
            if geometric is None:
                heuristic_labels += 1
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
            else:
                geometric_labels += 1
                progress, cost = geometric
                if action_id == next_action:
                    cost = min(cost, 0.45)
                if _is_repeated_ineffective(action_id, recent_actions):
                    cost = max(cost, 0.85)
            labels[str(action_id)] = {"progress": progress, "cost": cost}
            action_positive[str(action_id)] += progress
            action_cost[str(action_id)] += cost
        rows.append({"sample_id": row["sample_id"], "labels": labels})
    write_jsonl(args.output, rows)
    denom = max(1, len(windows))
    save_bar_png(diag_dir / "positive_progress_by_action.png", "Positive progress labels by action", action_positive)
    save_bar_png(diag_dir / "average_cost_by_action.png", "Average cost label by action", {k: v / denom for k, v in action_cost.items()})
    write_json(
        diag_dir / "summary.json",
        {
            "step_windows": len(windows),
            "label_rows": len(rows),
            "geometric_action_labels": geometric_labels,
            "heuristic_action_labels": heuristic_labels,
        },
    )
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


def _geometric_consequence_label(
    row: Dict,
    action_id: int,
    completion: float,
    milestone_changes: bool,
) -> Tuple[float, float] | None:
    current_pose = row.get("reference_pose")
    next_pose = row.get("next_reference_pose")
    if not current_pose or not next_pose or len(current_pose) < 7 or len(next_pose) < 3:
        return None

    curr_pos = np.asarray(current_pose[:3], dtype=np.float32)
    next_pos = np.asarray(next_pose[:3], dtype=np.float32)
    candidate_pos, candidate_yaw = _simulate_action(current_pose, action_id)
    segment = next_pos - curr_pos
    segment_len = float(np.linalg.norm(segment))
    if segment_len < 1e-6:
        segment = _heading_vector(_yaw_from_pose(next_pose))
        segment_len = 1.0

    displacement = candidate_pos - curr_pos
    progress_raw = float(np.dot(displacement, segment) / max(1e-6, segment_len * segment_len))
    target_distance = float(np.linalg.norm(candidate_pos - next_pos))
    current_distance = float(np.linalg.norm(curr_pos - next_pos))
    distance_gain = (current_distance - target_distance) / max(1e-6, current_distance)
    heading_gain = _heading_alignment(candidate_yaw, segment)

    if action_id == STOP:
        progress = 0.45 if completion >= 0.9 or milestone_changes else 0.0
        cost = 0.15 if completion >= 0.9 or milestone_changes else 1.0
        return progress, cost

    progress = np.clip(0.55 * max(0.0, progress_raw) + 0.35 * max(0.0, distance_gain) + 0.10 * heading_gain, 0.0, 1.0)
    off_route = target_distance / max(1.0, segment_len)
    reverse_penalty = max(0.0, -progress_raw)
    cost = np.clip(0.65 * off_route + 0.25 * reverse_penalty + 0.10 * (1.0 - heading_gain), 0.0, 1.0)
    return float(progress), float(cost)


def _simulate_action(pose: Iterable[float], action_id: int) -> Tuple[np.ndarray, float]:
    values = list(pose)
    position = np.asarray(values[:3], dtype=np.float32)
    yaw = _yaw_from_pose(values)

    if action_id == MOVE_FORWARD:
        return position + _heading_vector(yaw) * 5.0, yaw
    if action_id == TURN_LEFT:
        return position.copy(), _wrap_yaw(yaw - math.radians(15.0))
    if action_id == TURN_RIGHT:
        return position.copy(), _wrap_yaw(yaw + math.radians(15.0))
    if action_id == GO_UP:
        return position + np.asarray([0.0, 0.0, -2.0], dtype=np.float32), yaw
    if action_id == GO_DOWN:
        return position + np.asarray([0.0, 0.0, 2.0], dtype=np.float32), yaw
    if action_id == MOVE_LEFT:
        return position - _right_vector(yaw) * 5.0, yaw
    if action_id == MOVE_RIGHT:
        return position + _right_vector(yaw) * 5.0, yaw
    return position.copy(), yaw


def _yaw_from_pose(pose: Iterable[float]) -> float:
    values = list(pose)
    x, y, z, w = values[3], values[4], values[5], values[6]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_vector(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)


def _right_vector(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0], dtype=np.float32)


def _heading_alignment(yaw: float, target_vector: np.ndarray) -> float:
    norm = float(np.linalg.norm(target_vector[:2]))
    if norm < 1e-6:
        return 1.0
    target = target_vector[:2] / norm
    heading = _heading_vector(yaw)[:2]
    return float(np.clip((np.dot(heading, target) + 1.0) / 2.0, 0.0, 1.0))


def _wrap_yaw(yaw: float) -> float:
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw


if __name__ == "__main__":
    main()
