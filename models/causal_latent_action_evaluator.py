from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.action_encoder import ActionEncoder


class CausalLatentActionEvaluator(nn.Module):
    def __init__(
        self,
        num_actions: int,
        token_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_encoder = ActionEncoder(num_actions=num_actions, token_dim=token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.progress_head = _Head(token_dim)
        self.cost_head = _Head(token_dim)
        self.latent_norm = nn.LayerNorm(token_dim)

    def forward(self, state_dict: Dict[str, torch.Tensor], action_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_token = self.action_encoder(action_ids)
        tokens = torch.stack(
            [
                state_dict["current_visual_token"],
                state_dict["history_token"],
                state_dict["traj_token"],
                state_dict["milestone_token"],
                state_dict["progress_token"],
                action_token,
                state_dict["prev_latent"],
            ],
            dim=1,
        )
        seq_len = tokens.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        encoded = self.transformer(tokens, mask=causal_mask)
        next_latent = self.latent_norm(encoded[:, -1])
        return {
            "progress_gain": torch.sigmoid(self.progress_head(next_latent)).squeeze(-1),
            "cost": torch.sigmoid(self.cost_head(next_latent)).squeeze(-1),
            "next_latent": next_latent,
        }


class _Head(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

