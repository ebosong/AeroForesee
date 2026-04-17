from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int, token_dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_actions, token_dim)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(action_ids.long())

