from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import yaml


@dataclass
class FuserWeights:
    w_progress: float = 1.0
    w_cost: float = 0.7
    w_prior: float = 0.6


class DecisionFuser:
    def __init__(self, weights: Optional[FuserWeights] = None) -> None:
        self.weights = weights or FuserWeights()

    @classmethod
    def from_yaml(cls, path: str) -> "DecisionFuser":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(FuserWeights(
            w_progress=float(data.get("w_progress", 1.0)),
            w_cost=float(data.get("w_cost", 0.7)),
            w_prior=float(data.get("w_prior", 0.6)),
        ))

    def score(
        self,
        action_ids: Iterable[int],
        progress_gain: torch.Tensor,
        cost: torch.Tensor,
        prior: Dict[int, float],
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for idx, action_id in enumerate(action_ids):
            scores[int(action_id)] = (
                self.weights.w_progress * float(progress_gain[idx].detach().cpu())
                - self.weights.w_cost * float(cost[idx].detach().cpu())
                + self.weights.w_prior * float(prior.get(int(action_id), 0.0))
            )
        return scores

