from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from preprocess.common import read_jsonl


class V0ActionDataset(Dataset):
    def __init__(
        self,
        step_windows: str,
        rollout_labels: str,
        action_prior_cache: str | None = None,
        latent_index: str | None = None,
        image_height: int = 224,
        image_width: int = 224,
        max_keyframes: int = 8,
    ) -> None:
        self.windows = {row["sample_id"]: row for row in read_jsonl(step_windows)}
        label_rows = {row["sample_id"]: row["labels"] for row in read_jsonl(rollout_labels)}
        self.prior = _load_prior(action_prior_cache)
        self.latents = _load_latent_index(latent_index)
        self.items: List[tuple[str, int]] = []
        for sample_id, labels in label_rows.items():
            if sample_id not in self.windows:
                continue
            for action_id in labels:
                self.items.append((sample_id, int(action_id)))
        self.labels = label_rows
        self.image_height = image_height
        self.image_width = image_width
        self.max_keyframes = max_keyframes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id, action_id = self.items[index]
        row = self.windows[sample_id]
        label = self.labels[sample_id][str(action_id)]
        latent_path = self.latents.get(sample_id)
        latent_target = torch.load(latent_path, map_location="cpu") if latent_path else torch.zeros(512)
        return {
            "sample_id": sample_id,
            "current_rgb": torch.zeros(self.image_height, self.image_width, 3, dtype=torch.float32),
            "history_rgbs": torch.zeros(self.max_keyframes, self.image_height, self.image_width, 3, dtype=torch.float32),
            "action_history": torch.tensor(row["action_history"], dtype=torch.long),
            "pose_deltas": torch.tensor(row["pose_deltas"], dtype=torch.float32),
            "fallback_flags": torch.zeros(len(row["action_history"]), dtype=torch.float32),
            "milestone_id": torch.tensor(row["milestone_id"], dtype=torch.long),
            "milestone_text": row["milestone_text"],
            "completion": torch.tensor(float(row["completion"]), dtype=torch.float32),
            "recent_progress_flag": torch.tensor(float(row["recent_progress_flag"]), dtype=torch.float32),
            "prev_latent": torch.zeros_like(latent_target),
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "progress_label": torch.tensor(float(label["progress"]), dtype=torch.float32),
            "cost_label": torch.tensor(float(label["cost"]), dtype=torch.float32),
            "latent_target": latent_target.float(),
            "prior": torch.tensor(float(self.prior.get(sample_id, {}).get(str(action_id), 0.0)), dtype=torch.float32),
        }


def collate_v0(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tensor_keys = [
        "current_rgb", "history_rgbs", "action_history", "pose_deltas", "fallback_flags",
        "milestone_id", "completion", "recent_progress_flag", "prev_latent", "action_id",
        "progress_label", "cost_label", "latent_target", "prior",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["milestone_text"] = [item["milestone_text"] for item in batch]
    out["sample_id"] = [item["sample_id"] for item in batch]
    return out


def _load_prior(path: str | None) -> Dict[str, Dict[str, float]]:
    if not path or not Path(path).exists():
        return {}
    return {row["sample_id"]: row["prior"] for row in read_jsonl(path)}


def _load_latent_index(path: str | None) -> Dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    return {row["sample_id"]: row["latent_target"] for row in read_jsonl(path)}

