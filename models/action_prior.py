from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from PIL import Image

from models.action_space import AirVLNActionSpace
from models.vlm_clients import VLMClient, UniformVLMClient


class ActionPriorModule:
    def __init__(
        self,
        action_space: Optional[AirVLNActionSpace] = None,
        vlm_client: Optional[VLMClient] = None,
        prompt_path: str | Path = "prompts/action_prior_prompt.txt",
        stop_completion_threshold: float = 0.35,
    ) -> None:
        self.action_space = action_space or AirVLNActionSpace()
        self.vlm_client = vlm_client or UniformVLMClient()
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.stop_completion_threshold = stop_completion_threshold

    def score(
        self,
        current_rgb: Image.Image | np.ndarray | None,
        keyframes: List[Image.Image | np.ndarray],
        milestone_text: str,
        progress_summary: str,
        legal_action_ids: Optional[Iterable[int]] = None,
        milestone_completion: float = 0.0,
    ) -> Dict[int, float]:
        legal_ids = self.action_space.valid_ids(legal_action_ids)
        prompt_names = self.action_space.prompt_action_list(legal_ids)
        prompt = self._build_prompt(milestone_text, progress_summary, prompt_names)
        images = _to_pil_list([current_rgb] + list(keyframes[:2]))
        raw = self.vlm_client.score_actions(images, prompt, prompt_names)
        return self.postprocess(raw, legal_ids, milestone_completion)

    def postprocess(
        self,
        raw_scores: Mapping[str, float],
        legal_action_ids: Iterable[int],
        milestone_completion: float = 0.0,
    ) -> Dict[int, float]:
        legal_ids = self.action_space.valid_ids(legal_action_ids)
        scores: Dict[int, float] = {}
        for action_id in legal_ids:
            name = self.action_space.prompt_name(action_id)
            scores[action_id] = max(0.0, float(raw_scores.get(name, 0.0)))

        stop_id = self.action_space.id_from_official_name("STOP")
        if stop_id in scores and milestone_completion < self.stop_completion_threshold:
            scores[stop_id] = min(scores[stop_id], 0.05)

        total = sum(scores.values())
        if total <= 0:
            value = 1.0 / max(1, len(scores))
            return {k: value for k in scores}
        return {k: v / total for k, v in scores.items()}

    def _build_prompt(self, milestone_text: str, progress_summary: str, action_names: List[str]) -> str:
        return (
            f"{self.prompt_template}\n\n"
            f"Current milestone: {milestone_text}\n"
            f"Current progress summary: {progress_summary}\n"
            f"Official action list: {json.dumps(action_names)}\n"
        )


def _to_pil_list(items: List[Image.Image | np.ndarray | None]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
        else:
            arr = np.asarray(item)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))
    return images

