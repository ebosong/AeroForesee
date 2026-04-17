from __future__ import annotations

import torch
from torch import nn


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        num_actions: int,
        token_dim: int = 512,
        action_dim: int = 32,
        pose_dim: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions + 1, action_dim, padding_idx=num_actions)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim + 1, 64), nn.ReLU(inplace=True))
        self.gru = nn.GRU(action_dim + 64, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, token_dim)
        self.num_actions = num_actions

    def forward(
        self,
        action_history: torch.Tensor,
        pose_deltas: torch.Tensor,
        fallback_flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_history.ndim == 1:
            action_history = action_history.unsqueeze(0)
        if pose_deltas.ndim == 2:
            pose_deltas = pose_deltas.unsqueeze(0)
        action_history = action_history.clamp(min=0, max=self.num_actions).long()
        if fallback_flags is None:
            fallback_flags = torch.zeros(action_history.shape, dtype=pose_deltas.dtype, device=pose_deltas.device)
        if fallback_flags.ndim == 1:
            fallback_flags = fallback_flags.unsqueeze(0)
        action_token = self.action_embedding(action_history)
        pose_input = torch.cat([pose_deltas.float(), fallback_flags.float().unsqueeze(-1)], dim=-1)
        pose_token = self.pose_mlp(pose_input)
        x = torch.cat([action_token, pose_token], dim=-1)
        _, h = self.gru(x)
        return self.out(h[-1])

