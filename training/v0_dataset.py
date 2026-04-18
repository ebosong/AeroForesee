from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
        image_root: str | None = None,
        latent_dim: int = 512,
    ) -> None:
        self.windows = {row["sample_id"]: row for row in read_jsonl(step_windows)}
        label_rows = {row["sample_id"]: row["labels"] for row in read_jsonl(rollout_labels)}
        self.prior = _load_prior(action_prior_cache)
        self.latents = _load_latent_index(latent_index)
        self.prev_latents = self._build_prev_latent_index()
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
        self.image_root = Path(image_root) if image_root else None
        self.latent_dim = latent_dim

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id, action_id = self.items[index]
        row = self.windows[sample_id]
        label = self.labels[sample_id][str(action_id)]
        latent_path = self.latents.get(sample_id)
        latent_target = _load_latent(latent_path, latent_dim=self.latent_dim)
        prev_latent = _load_latent(self.prev_latents.get(sample_id), fallback=torch.zeros_like(latent_target))
        history = _load_history_rgbs(
            row.get("keyframe_rgb_paths") or [],
            self.max_keyframes,
            self.image_height,
            self.image_width,
            self.image_root,
        )
        return {
            "sample_id": sample_id,
            "current_rgb": _load_rgb(row.get("rgb_path"), self.image_height, self.image_width, self.image_root),
            "history_rgbs": history,
            "action_history": torch.tensor(row["action_history"], dtype=torch.long),
            "pose_deltas": torch.tensor(row["pose_deltas"], dtype=torch.float32),
            "fallback_flags": torch.zeros(len(row["action_history"]), dtype=torch.float32),
            "milestone_id": torch.tensor(row["milestone_id"], dtype=torch.long),
            "milestone_text": row["milestone_text"],
            "completion": torch.tensor(float(row["completion"]), dtype=torch.float32),
            "recent_progress_flag": torch.tensor(float(row["recent_progress_flag"]), dtype=torch.float32),
            "prev_latent": prev_latent.float(),
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "progress_label": torch.tensor(float(label["progress"]), dtype=torch.float32),
            "cost_label": torch.tensor(float(label["cost"]), dtype=torch.float32),
            "latent_target": latent_target.float(),
            "prior": torch.tensor(float(self.prior.get(sample_id, {}).get(str(action_id), 0.0)), dtype=torch.float32),
        }

    def _build_prev_latent_index(self) -> Dict[str, str]:
        prev_index: Dict[str, str] = {}
        for sample_id, row in self.windows.items():
            prev_sample_id = row.get("prev_sample_id")
            if not prev_sample_id and int(row.get("t", 0)) > 0:
                prev_sample_id = f"{row.get('instruction_id', '')}_{int(row['t']) - 1}"
            if prev_sample_id in self.latents:
                prev_index[sample_id] = self.latents[prev_sample_id]
        return prev_index


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


def _load_latent(path: str | None, fallback: torch.Tensor | None = None, latent_dim: int = 512) -> torch.Tensor:
    if path:
        try:
            return torch.load(path, map_location="cpu").float()
        except Exception:
            pass
    if fallback is not None:
        return fallback.float()
    return torch.zeros(latent_dim, dtype=torch.float32)


def _resolve_image_path(path: str | None, image_root: Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() and image_root is not None:
        candidate = image_root / candidate
    return candidate if candidate.exists() else None


def _load_rgb(path: str | None, height: int, width: int, image_root: Path | None) -> torch.Tensor:
    resolved = _resolve_image_path(path, image_root)
    if resolved is None:
        return torch.zeros(height, width, 3, dtype=torch.float32)
    try:
        from PIL import Image

        image = Image.open(resolved).convert("RGB").resize((width, height))
        return torch.tensor(np.asarray(image), dtype=torch.float32)
    except Exception:
        return torch.zeros(height, width, 3, dtype=torch.float32)


def _load_history_rgbs(paths: List[str], max_keyframes: int, height: int, width: int, image_root: Path | None) -> torch.Tensor:
    frames = [_load_rgb(path, height, width, image_root) for path in paths[-max_keyframes:]]
    while len(frames) < max_keyframes:
        frames.insert(0, torch.zeros(height, width, 3, dtype=torch.float32))
    return torch.stack(frames[-max_keyframes:], dim=0)

