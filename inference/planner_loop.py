from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from models.action_prior import ActionPriorModule
from models.action_space import AirVLNActionSpace
from models.causal_latent_action_evaluator import CausalLatentActionEvaluator
from models.decision_fuser import DecisionFuser
from models.fallback import FallbackPolicy
from models.state_builder import MilestoneAwareStateBuilder
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_svg


@dataclass
class EpisodeMemory:
    rgb_history: List[np.ndarray] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)
    pose_history: List[np.ndarray] = field(default_factory=list)
    fallback_history: List[float] = field(default_factory=list)
    latent: Optional[torch.Tensor] = None
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)


class V0PlannerLoop:
    def __init__(
        self,
        state_builder: MilestoneAwareStateBuilder,
        evaluator: CausalLatentActionEvaluator,
        action_prior: Optional[ActionPriorModule] = None,
        fuser: Optional[DecisionFuser] = None,
        device: torch.device | str = "cpu",
        history_len: int = 16,
        max_keyframes: int = 8,
        fallback_config: Optional[Mapping[str, Any]] = None,
        diagnostics_dir: str = "DATA/v0/diagnostics/planner_loop",
    ) -> None:
        self.device = torch.device(device)
        self.action_space = AirVLNActionSpace()
        self.state_builder = state_builder.to(self.device).eval()
        self.evaluator = evaluator.to(self.device).eval()
        self.action_prior = action_prior or ActionPriorModule(action_space=self.action_space)
        self.fuser = fuser or DecisionFuser()
        self.history_len = history_len
        self.max_keyframes = max_keyframes
        self.fallback_config = dict(fallback_config or getattr(self.fuser, "fallback_config", {}) or {})
        self.memories: Dict[int, EpisodeMemory] = {}
        self.diagnostics_dir = ensure_dir(diagnostics_dir)

    @torch.no_grad()
    def act(self, observations: List[Dict[str, Any]], batch_items: List[Dict[str, Any]], step: int) -> Dict[str, Any]:
        self._ensure_memories(batch_items)
        rgbs = []
        histories = []
        action_histories = []
        pose_deltas = []
        fallback_flags = []
        milestone_ids = []
        milestone_texts = []
        completions = []
        recent_progress = []
        prev_latents = []
        priors = []

        for index, (obs, item) in enumerate(zip(observations, batch_items)):
            episode_id = int(item["episode_id"])
            memory = self.memories[episode_id]
            rgb = np.asarray(obs.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8)))
            pose = np.asarray(obs.get("pose", np.zeros(7)), dtype=np.float32)
            progress = float(obs.get("progress", 0.0))
            instruction = item.get("instruction", {}).get("instruction_text", "")
            milestone_text, milestone_id, completion = _milestone_from_progress(item, progress)
            if not milestone_text:
                milestone_text = instruction
            rgbs.append(_resize_like_training(rgb))
            histories.append(_pad_history(memory.rgb_history[-self.max_keyframes:], self.max_keyframes))
            action_histories.append(_pad_ints(memory.action_history[-self.history_len:], self.history_len))
            pose_deltas.append(_pose_deltas(memory.pose_history[-self.history_len:], pose, self.history_len))
            fallback_flags.append(_pad_floats(memory.fallback_history[-self.history_len:], self.history_len))
            milestone_ids.append(milestone_id)
            milestone_texts.append(milestone_text)
            completions.append(completion)
            recent_progress.append(float(progress > 0.0))
            prev_latents.append(memory.latent if memory.latent is not None else torch.zeros(self.state_builder.token_dim))
            priors.append(self.action_prior.score(
                current_rgb=rgb,
                keyframes=memory.rgb_history[-2:],
                milestone_text=milestone_text,
                progress_summary=f"completion={completion:.3f}",
                legal_action_ids=self.action_space.action_ids,
                milestone_completion=completion,
            ))
            memory.rgb_history.append(rgb)
            memory.pose_history.append(pose)

        state = self.state_builder(
            current_rgb=torch.tensor(np.stack(rgbs), dtype=torch.float32, device=self.device),
            history_rgbs=torch.tensor(np.stack(histories), dtype=torch.float32, device=self.device),
            action_history=torch.tensor(action_histories, dtype=torch.long, device=self.device),
            pose_deltas=torch.tensor(pose_deltas, dtype=torch.float32, device=self.device),
            milestone_ids=torch.tensor(milestone_ids, dtype=torch.long, device=self.device),
            milestone_texts=milestone_texts,
            completion=torch.tensor(completions, dtype=torch.float32, device=self.device),
            recent_progress_flag=torch.tensor(recent_progress, dtype=torch.float32, device=self.device),
            prev_latent=torch.stack([latent.to(self.device) for latent in prev_latents], dim=0),
            fallback_flags=torch.tensor(fallback_flags, dtype=torch.float32, device=self.device),
        )

        per_action_outputs = {}
        for action_id in self.action_space.action_ids:
            action_tensor = torch.full((len(observations),), action_id, dtype=torch.long, device=self.device)
            per_action_outputs[action_id] = self.evaluator(state, action_tensor)

        actions: List[int] = []
        fallbacks: List[bool] = []
        scores: List[Dict[int, float]] = []
        for batch_idx, item in enumerate(batch_items):
            progress_gain = torch.stack([per_action_outputs[a]["progress_gain"][batch_idx] for a in self.action_space.action_ids])
            cost = torch.stack([per_action_outputs[a]["cost"][batch_idx] for a in self.action_space.action_ids])
            fused = self.fuser.score(self.action_space.action_ids, progress_gain, cost, priors[batch_idx])
            episode_id = int(item["episode_id"])
            chosen, fallback = self.memories[episode_id].fallback.select(fused, float(progress_gain.max().detach().cpu()))
            self.memories[episode_id].latent = per_action_outputs[chosen]["next_latent"][batch_idx].detach().cpu()
            self.memories[episode_id].action_history.append(chosen)
            self.memories[episode_id].fallback_history.append(float(fallback))
            actions.append(chosen)
            fallbacks.append(fallback)
            scores.append(fused)
            readable_scores = {self.action_space.official_name(k): v for k, v in fused.items()}
            print_event(
                "planner_loop",
                "step_decision",
                episode_id=episode_id,
                step=step,
                action=self.action_space.official_name(chosen),
                fallback=fallback,
                best_score=f"{max(fused.values()):.4f}",
            )
            append_jsonl(
                self.diagnostics_dir / "step_decisions.jsonl",
                {
                    "episode_id": episode_id,
                    "step": step,
                    "action_id": chosen,
                    "action_name": self.action_space.official_name(chosen),
                    "fallback": fallback,
                    "scores": readable_scores,
                },
            )
            if step < 5:
                save_bar_svg(
                    self.diagnostics_dir / f"episode_{episode_id}_step_{step}_scores.svg",
                    f"Episode {episode_id} step {step} fused scores",
                    readable_scores,
                )
        return {"actions": actions, "fallbacks": fallbacks, "scores": scores}

    def _ensure_memories(self, batch_items: List[Dict[str, Any]]) -> None:
        active = {int(item["episode_id"]) for item in batch_items}
        for episode_id in active:
            if episode_id not in self.memories:
                self.memories[episode_id] = EpisodeMemory(
                    fallback=FallbackPolicy(
                        action_space=self.action_space,
                        low_progress_threshold=float(self.fallback_config.get("low_progress_threshold", 0.05)),
                        low_score_threshold=float(self.fallback_config.get("low_score_threshold", 0.05)),
                        progress_patience=int(self.fallback_config.get("progress_patience", 3)),
                        repeated_turn_patience=int(self.fallback_config.get("repeated_turn_patience", 2)),
                        conservative_actions=self.fallback_config.get("conservative_actions", ["MOVE_FORWARD", "STOP"]),
                    )
                )


def _resize_like_training(rgb: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((width, height)))
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def _pad_history(history: List[np.ndarray], length: int) -> np.ndarray:
    frames = [_resize_like_training(frame) for frame in history[-length:]]
    while len(frames) < length:
        frames.insert(0, np.zeros((224, 224, 3), dtype=np.uint8))
    return np.stack(frames, axis=0)


def _pad_ints(values: List[int], length: int) -> List[int]:
    values = list(values[-length:])
    while len(values) < length:
        values.insert(0, 0)
    return values


def _pad_floats(values: List[float], length: int) -> List[float]:
    values = list(values[-length:])
    while len(values) < length:
        values.insert(0, 0.0)
    return values


def _pose_deltas(history: List[np.ndarray], current_pose: np.ndarray, length: int) -> List[List[float]]:
    poses = list(history[-length:]) + [current_pose]
    deltas: List[List[float]] = []
    for idx in range(1, len(poses)):
        curr = np.asarray(poses[idx][:3], dtype=np.float32)
        prev = np.asarray(poses[idx - 1][:3], dtype=np.float32)
        deltas.append([float(v) for v in (curr - prev).tolist()] + [0.0])
    while len(deltas) < length:
        deltas.insert(0, [0.0, 0.0, 0.0, 0.0])
    return deltas[-length:]


def _milestone_from_progress(item: Dict[str, Any], progress: float) -> tuple[str, int, float]:
    milestones = item.get("milestones") or item.get("instruction_plan") or []
    if not milestones:
        return "", 1, progress
    idx = max(0, min(len(milestones) - 1, int(progress * len(milestones))))
    milestone = milestones[idx]
    text = (
        f"{milestone.get('action_type', '')} {milestone.get('spatial_relation', '')} "
        f"{', '.join(milestone.get('landmarks', []))}"
    ).strip()
    local_start = idx / max(1, len(milestones))
    local_end = (idx + 1) / max(1, len(milestones))
    local_completion = (progress - local_start) / max(1e-6, local_end - local_start)
    return text, int(milestone.get("mid", idx + 1)), float(np.clip(local_completion, 0.0, 1.0))
