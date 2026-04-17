from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_svg, write_json


def load_plans(path: str | Path) -> Dict[str, Dict[str, Any]]:
    if not Path(path).exists():
        return {}
    return {row["instruction_id"]: row for row in read_jsonl(path)}


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
    rows = []
    step_hist: Dict[str, float] = {}
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
            start = max(0, t - args.history_len)
            action_history = actions[start:t]
            deltas = [pose_delta(ref_path, i) for i in range(start, t)] if ref_path else []
            while len(action_history) < args.history_len:
                action_history.insert(0, 0)
                deltas.insert(0, [0.0, 0.0, 0.0, 0.0])
            keyframes = select_keyframe_indices(t, actions, milestone_ids, progress_values, args.max_keyframes, args.keyframe_interval)
            rows.append({
                "sample_id": f"{item_id}_{t}",
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "t": t,
                "instruction_id": item_id,
                "milestone_id": mid,
                "milestone": milestone,
                "milestone_text": milestone_to_text(milestone),
                "completion": progress_values[t],
                "recent_progress_flag": float(t > 0 and progress_values[t] > progress_values[t - 1]),
                "gt_action": actions[t],
                "next_action": actions[t + 1] if t + 1 < len(actions) else 0,
                "action_history": action_history[-args.history_len:],
                "pose_deltas": deltas[-args.history_len:],
                "keyframe_indices": keyframes,
                "legal_action_ids": list(range(8)),
                "reference_position": ref_path[t][0:3] if ref_path and t < len(ref_path) else None,
                "next_reference_position": ref_path[t + 1][0:3] if ref_path and t + 1 < len(ref_path) else None,
            })
        step_hist[str(min(500, steps))] = step_hist.get(str(min(500, steps)), 0.0) + 1.0
        print_event("build_step_windows", "episode_done", instruction_id=item_id, steps=steps, milestones=len(milestones))
        append_jsonl(diag_dir / "events.jsonl", {"stage": "build_step_windows", "instruction_id": item_id, "steps": steps, "milestones": len(milestones)})
    write_jsonl(args.output, rows)
    write_json(diag_dir / "summary.json", {"episodes": len(episodes), "step_windows": len(rows)})
    save_bar_svg(diag_dir / "episode_step_count_distribution.svg", "Episode step count distribution", step_hist)
    print_event("build_step_windows", "done", step_windows=len(rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--instruction-plan", default="data/instruction_plan.jsonl")
    parser.add_argument("--output", default="data/step_windows/train.jsonl")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=4)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_step_windows")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
