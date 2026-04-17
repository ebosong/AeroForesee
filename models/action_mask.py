from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from models.action_space import AirVLNActionSpace


class ActionMask:
    def __init__(self, action_space: Optional[AirVLNActionSpace] = None) -> None:
        self.action_space = action_space or AirVLNActionSpace()

    def legal_ids(self, legal_action_ids: Optional[Iterable[int]] = None) -> List[int]:
        return self.action_space.valid_ids(legal_action_ids)

    def tensor_mask(self, legal_action_ids: Optional[Iterable[int]], device: torch.device | str = "cpu") -> torch.Tensor:
        mask = torch.zeros(self.action_space.num_actions, dtype=torch.bool, device=device)
        for action_id in self.legal_ids(legal_action_ids):
            mask[action_id] = True
        return mask

    def apply(self, scores: torch.Tensor, legal_action_ids: Optional[Iterable[int]]) -> torch.Tensor:
        mask = self.tensor_mask(legal_action_ids, scores.device)
        return scores.masked_fill(~mask, float("-inf"))

