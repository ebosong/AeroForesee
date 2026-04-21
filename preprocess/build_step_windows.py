from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.history_encoder import select_keyframe_indices
from preprocess.common import (
    instruction_id,
    load_episodes,
    milestone_to_text,
    nearest_milestone_index,
    pose_delta,
    read_jsonl,
    write_jsonl,
)
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_png, write_json


def load_plans(path: str | Path) -> Dict[str, Dict[str, Any]]:
    if not Path(path).exists():
        return {}
    return {row["instruction_id"]: row for row in read_jsonl(path)}


def load_rgb_index(path: str | Path | None) -> Dict[str, Dict[int, str]]:
    if not path or not Path(path).exists():
        return {}
    index: Dict[str, Dict[int, str]] = {}
    for row in read_jsonl(path):
        step = int(row["step"])
        image_path = str(row["rgb_path"])
        keys = [
            row.get("trajectory_id"),
            row.get("episode_id"),
            row.get("instruction_id"),
        ]
        for key in keys:
            if key is None:
                continue
            index.setdefault(str(key), {})[step] = image_path
    return index


def default_plan(item_id: str, instruction: str) -> Dict[str, Any]:
    words = instruction.split()
    chunks = [" ".join(words[i::3]) for i in range(3)]
    return {
        "instruction_id": item_id,
        "milestones": [
            {
                "mid": i + 1,
                "action_type": "follow",
                "landmarks": [chunk or "route"],
                "spatial_relation": "toward",
                "verification_cues": ["progress along reference route"],
            }
            for i, chunk in enumerate(chunks)
        ],
    }


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    episodes = load_episodes(args.dataset_json)
    plans = load_plans(args.instruction_plan)
    rgb_index = load_rgb_index(args.rgb_index)
    rows = []
    step_hist: Dict[str, float] = {}
    indexed_rgb = 0
    missing_rgb = 0
    print_event("build_step_windows", "start", dataset=args.dataset_json, episodes=len(episodes))
    for episode in episodes:
        item_id = instruction_id(episode)
        text = episode.get("instruction", {}).get("instruction_text", "")
        plan = plans.get(item_id) or default_plan(item_id, text)
        milestones = plan["milestones"]
        actions = [int(a) for a in episode.get("actions", [])]
        ref_path = episode.get("reference_path", [])
        steps = min(len(actions), len(ref_path) - 1 if ref_path else len(actions))
        progress_values = [i / max(1, steps) for i in range(steps + 1)]
        milestone_ids = [nearest_milestone_index(progress_values[i], len(milestones)) for i in range(steps + 1)]
        for t in range(steps):
            mid = milestone_ids[t]
            milestone = milestones[mid - 1]
            local_completion = _local_completion(progress_values[t], mid, len(milestones))
            next_mid = milestone_ids[t + 1] if t + 1 < len(milestone_ids) else mid
            start = max(0, t - args.history_len)
            action_history = actions[start:t]
            deltas = [pose_delta(ref_path, i) for i in range(start, t)] if ref_path else []
            while len(action_history) < args.history_len:
                action_history.insert(0, 0)
                deltas.insert(0, [0.0, 0.0, 0.0, 0.0])
            keyframes = select_keyframe_indices(t, actions, milestone_ids, progress_values, args.max_keyframes, args.keyframe_interval)
            rgb_path = _path_at(episode, ["rgb_paths", "image_paths", "images", "frames"], t) or _indexed_rgb_path(rgb_index, episode, item_id, t)
            next_rgb_path = _path_at(episode, ["rgb_paths", "image_paths", "images", "frames"], t + 1) or _indexed_rgb_path(rgb_index, episode, item_id, t + 1)
            keyframe_rgb_paths = [
                _path_at(episode, ["rgb_paths", "image_paths", "images", "frames"], idx) or _indexed_rgb_path(rgb_index, episode, item_id, idx)
                for idx in keyframes
            ]
            keyframe_rgb_paths = [path for path in keyframe_rgb_paths if path]
            indexed_rgb += int(bool(rgb_path and not _path_at(episode, ["rgb_paths", "image_paths", "images", "frames"], t)))
            missing_rgb += int(not bool(rgb_path))
            rows.append({
                "sample_id": f"{item_id}_{t}",
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "t": t,
                "prev_sample_id": f"{item_id}_{t - 1}" if t > 0 else None,
                "next_sample_id": f"{item_id}_{t + 1}" if t + 1 < steps else None,
                "instruction_id": item_id,
                "milestone_id": mid,
                "next_milestone_id": next_mid,
                "milestone": milestone,
                "milestone_text": milestone_to_text(milestone),
                "completion": local_completion,
                "milestone_completion": local_completion,
                "global_completion": progress_values[t],
                "next_global_completion": progress_values[t + 1] if t + 1 < len(progress_values) else progress_values[t],
                "recent_progress_flag": float(t > 0 and progress_values[t] > progress_values[t - 1]),
                "gt_action": actions[t],
                "next_action": actions[t + 1] if t + 1 < len(actions) else 0,
                "action_history": action_history[-args.history_len:],
                "pose_deltas": deltas[-args.history_len:],
                "keyframe_indices": keyframes,
                "rgb_path": rgb_path,
                "next_rgb_path": next_rgb_path,
                "keyframe_rgb_paths": keyframe_rgb_paths,
                "legal_action_ids": list(range(8)),
                "reference_pose": ref_path[t] if ref_path and t < len(ref_path) else None,
                "next_reference_pose": ref_path[t + 1] if ref_path and t + 1 < len(ref_path) else None,
                "prev_reference_pose": ref_path[t - 1] if ref_path and t > 0 else None,
                "reference_position": ref_path[t][0:3] if ref_path and t < len(ref_path) else None,
                "next_reference_position": ref_path[t + 1][0:3] if ref_path and t + 1 < len(ref_path) else None,
            })
        step_hist[str(min(500, steps))] = step_hist.get(str(min(500, steps)), 0.0) + 1.0
        print_event("build_step_windows", "episode_done", instruction_id=item_id, steps=steps, milestones=len(milestones))
        append_jsonl(diag_dir / "events.jsonl", {"stage": "build_step_windows", "instruction_id": item_id, "steps": steps, "milestones": len(milestones)})
    write_jsonl(args.output, rows)
    write_json(
        diag_dir / "summary.json",
        {
            "episodes": len(episodes),
            "step_windows": len(rows),
            "rgb_index": args.rgb_index,
            "rgb_from_index": indexed_rgb,
            "missing_rgb_path": missing_rgb,
        },
    )
    save_bar_png(diag_dir / "episode_step_count_distribution.png", "Episode step count distribution", step_hist)
    print_event("build_step_windows", "done", step_windows=len(rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--instruction-plan", default="data/instruction_plan.jsonl")
    parser.add_argument("--rgb-index", default=None, help="JSONL produced by preprocess/export_lmdb_rgb.py; used when annotation episodes do not contain image paths.")
    parser.add_argument("--output", default="data/step_windows/train.jsonl")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=4)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_step_windows")
    build(parser.parse_args())


def _path_at(episode: Dict[str, Any], keys: List[str], index: int) -> str | None:
    for key in keys:
        values = episode.get(key)
        if isinstance(values, list) and 0 <= index < len(values):
            value = values[index]
            if isinstance(value, dict):
                value = value.get("path") or value.get("rgb") or value.get("image")
            if value:
                return str(value)
    return None


def _indexed_rgb_path(index: Dict[str, Dict[int, str]], episode: Dict[str, Any], item_id: str, step: int) -> str | None:
    for key in (episode.get("trajectory_id"), episode.get("episode_id"), item_id):
        if key is None:
            continue
        path = index.get(str(key), {}).get(int(step))
        if path:
            return path
    return None


def _local_completion(global_progress: float, milestone_id: int, num_milestones: int) -> float:
    if num_milestones <= 0:
        return float(global_progress)
    local_start = (milestone_id - 1) / num_milestones
    local_end = milestone_id / num_milestones
    return float(np.clip((global_progress - local_start) / max(1e-6, local_end - local_start), 0.0, 1.0))


if __name__ == "__main__":
    main()
