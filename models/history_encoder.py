from __future__ import annotations

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(self, token_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, current_visual_token: torch.Tensor, history_frame_tokens: torch.Tensor | None) -> torch.Tensor:
        if history_frame_tokens is None or history_frame_tokens.numel() == 0:
            return torch.zeros_like(current_visual_token)
        if history_frame_tokens.ndim == 2:
            history_frame_tokens = history_frame_tokens.unsqueeze(1)
        query = current_visual_token.unsqueeze(1)
        output, _ = self.attn(query, history_frame_tokens, history_frame_tokens)
        return self.norm(output.squeeze(1))


def select_keyframe_indices(
    step: int,
    actions: list[int],
    milestone_ids: list[int],
    progress_values: list[float],
    max_keyframes: int = 8,
    interval: int = 4,
) -> list[int]:
    selected = set(range(0, max(0, step), interval))
    for idx in range(1, min(step, len(milestone_ids))):
        if milestone_ids[idx] != milestone_ids[idx - 1]:
            selected.add(max(0, idx - 1))
            selected.add(idx)
    for idx in range(1, min(step, len(actions))):
        if actions[idx] in (2, 3) and actions[idx - 1] in (2, 3):
            selected.add(idx)
    for idx in range(3, min(step, len(progress_values))):
        gains = [progress_values[idx - j] - progress_values[idx - j - 1] for j in range(3)]
        if all(gain <= 1e-4 for gain in gains):
            selected.add(max(0, idx - 1))
            selected.add(idx)
    selected = sorted(i for i in selected if i < step)
    return selected[-max_keyframes:]

