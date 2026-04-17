from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_episodes(dataset_json: str | Path) -> List[Dict[str, Any]]:
    data = read_json(dataset_json)
    return data["episodes"] if isinstance(data, dict) and "episodes" in data else data


def instruction_id(episode: Dict[str, Any]) -> str:
    return str(episode.get("episode_id", episode.get("trajectory_id", "")))


def instruction_text(episode: Dict[str, Any]) -> str:
    instruction = episode.get("instruction", {})
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", ""))
    return str(instruction)


def pose_delta(path: List[List[float]], t: int) -> List[float]:
    if t <= 0 or t >= len(path):
        return [0.0, 0.0, 0.0, 0.0]
    curr = np.array(path[t][0:3], dtype=np.float32)
    prev = np.array(path[t - 1][0:3], dtype=np.float32)
    yaw_delta = 0.0
    return [float(v) for v in (curr - prev).tolist()] + [yaw_delta]


def nearest_milestone_index(progress: float, num_milestones: int) -> int:
    if num_milestones <= 0:
        return 1
    idx = int(progress * num_milestones) + 1
    return max(1, min(num_milestones, idx))


def milestone_to_text(milestone: Dict[str, Any]) -> str:
    landmarks = ", ".join(milestone.get("landmarks", []))
    cues = "; ".join(milestone.get("verification_cues", []))
    return (
        f"{milestone.get('action_type', '')} "
        f"{milestone.get('spatial_relation', '')} "
        f"{landmarks}. cues: {cues}"
    ).strip()

