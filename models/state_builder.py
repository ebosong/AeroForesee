from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from models.action_space import AirVLNActionSpace
from models.history_encoder import HistoryEncoder
from models.trajectory_encoder import TrajectoryEncoder
from models.vision_backbone import VisionBackbone


class HashTextEncoder(nn.Module):
    def __init__(self, token_dim: int = 512, vocab_size: int = 4096) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim)
        self.vocab_size = vocab_size
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, texts: list[str]) -> torch.Tensor:
        rows = []
        device = self.embedding.weight.device
        for text in texts:
            words = [w for w in text.lower().replace(",", " ").split() if w]
            if not words:
                words = ["empty"]
            ids = torch.tensor([hash(w) % self.vocab_size for w in words[:64]], dtype=torch.long, device=device)
            rows.append(self.embedding(ids).mean(dim=0))
        return self.norm(torch.stack(rows, dim=0))


class MilestoneAwareStateBuilder(nn.Module):
    def __init__(
        self,
        token_dim: int = 512,
        max_milestones: int = 8,
        action_space: Optional[AirVLNActionSpace] = None,
        vision_backbone: str = "dinov2_s",
        vision_pretrained: bool = False,
        vision_freeze: bool = True,
    ) -> None:
        super().__init__()
        self.action_space = action_space or AirVLNActionSpace()
        self.vision_backbone = VisionBackbone(
            token_dim=token_dim,
            backbone=vision_backbone,
            pretrained=vision_pretrained,
            freeze=vision_freeze,
        )
        self.history_encoder = HistoryEncoder(token_dim=token_dim)
        self.trajectory_encoder = TrajectoryEncoder(num_actions=self.action_space.num_actions, token_dim=token_dim)
        self.milestone_id_embedding = nn.Embedding(max_milestones + 1, token_dim)
        self.text_encoder = HashTextEncoder(token_dim=token_dim)
        self.progress_mlp = nn.Sequential(nn.Linear(2, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.token_dim = token_dim

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.vision_backbone(rgb)[0]

    def forward(
        self,
        current_rgb: torch.Tensor,
        history_rgbs: torch.Tensor | None,
        action_history: torch.Tensor,
        pose_deltas: torch.Tensor,
        milestone_ids: torch.Tensor,
        milestone_texts: list[str],
        completion: torch.Tensor,
        recent_progress_flag: torch.Tensor,
        prev_latent: torch.Tensor | None = None,
        fallback_flags: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        current_visual_token, _ = self.vision_backbone(current_rgb)
        history_frame_tokens = None
        if history_rgbs is not None and history_rgbs.numel() > 0:
            batch, frames = history_rgbs.shape[:2]
            flat = history_rgbs.reshape(batch * frames, *history_rgbs.shape[2:])
            history_frame_tokens = self.vision_backbone(flat)[0].reshape(batch, frames, -1)
        history_token = self.history_encoder(current_visual_token, history_frame_tokens)
        traj_token = self.trajectory_encoder(action_history, pose_deltas, fallback_flags)
        text_token = self.text_encoder(milestone_texts)
        milestone_token = self.milestone_id_embedding(milestone_ids.long().clamp(min=0, max=self.milestone_id_embedding.num_embeddings - 1)) + text_token
        progress_input = torch.stack([completion.float(), recent_progress_flag.float()], dim=-1)
        progress_token = self.progress_mlp(progress_input)
        if prev_latent is None:
            prev_latent = torch.zeros_like(current_visual_token)
        return {
            "current_visual_token": current_visual_token,
            "history_token": history_token,
            "traj_token": traj_token,
            "milestone_token": milestone_token,
            "progress_token": progress_token,
            "prev_latent": prev_latent,
        }

