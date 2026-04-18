from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
GO_UP = 4
GO_DOWN = 5
MOVE_LEFT = 6
MOVE_RIGHT = 7


@dataclass
class MilestoneStatus:
    text: str
    milestone_id: int
    completion: float
    index: int


class MilestoneProgressController:
    def __init__(
        self,
        advance_completion: float = 0.92,
        stop_prior_threshold: float = 0.35,
        min_steps_per_milestone: int = 2,
    ) -> None:
        self.active_index = 0
        self.completion = 0.0
        self.steps_in_milestone = 0
        self.advance_completion = advance_completion
        self.stop_prior_threshold = stop_prior_threshold
        self.min_steps_per_milestone = min_steps_per_milestone

    def current(self, item: Mapping[str, Any], global_progress: float) -> MilestoneStatus:
        milestones = _milestones(item)
        if not milestones:
            return MilestoneStatus("", 1, float(global_progress), 0)
        progress_index, progress_completion = _from_global_progress(global_progress, len(milestones))
        if global_progress > 0.0 and progress_index > self.active_index:
            self.active_index = progress_index
            self.completion = progress_completion
            self.steps_in_milestone = 0
        elif progress_index == self.active_index:
            self.completion = max(self.completion, progress_completion)
        self.active_index = max(0, min(self.active_index, len(milestones) - 1))
        milestone = milestones[self.active_index]
        return MilestoneStatus(
            text=_milestone_text(milestone),
            milestone_id=int(milestone.get("mid", self.active_index + 1)),
            completion=float(np.clip(self.completion, 0.0, 1.0)),
            index=self.active_index,
        )

    def update(
        self,
        item: Mapping[str, Any],
        chosen_action: int,
        prior: Mapping[int, float],
        progress_gain: float,
        global_progress: float,
    ) -> MilestoneStatus:
        milestones = _milestones(item)
        if not milestones:
            return MilestoneStatus("", 1, float(global_progress), 0)

        _, progress_completion = _from_global_progress(global_progress, len(milestones))
        self.completion = max(self.completion, progress_completion)
        stop_prior = float(prior.get(STOP, 0.0))
        action_increment = _action_increment(chosen_action)
        self.completion = float(np.clip(self.completion + action_increment + 0.25 * max(0.0, progress_gain), 0.0, 1.0))
        if stop_prior >= self.stop_prior_threshold:
            self.completion = max(self.completion, min(1.0, 0.80 + 0.25 * stop_prior))

        self.steps_in_milestone += 1
        if self._should_advance(chosen_action, stop_prior) and self.active_index < len(milestones) - 1:
            self.active_index += 1
            self.completion = 0.0
            self.steps_in_milestone = 0

        milestone = milestones[self.active_index]
        return MilestoneStatus(
            text=_milestone_text(milestone),
            milestone_id=int(milestone.get("mid", self.active_index + 1)),
            completion=float(np.clip(self.completion, 0.0, 1.0)),
            index=self.active_index,
        )

    def _should_advance(self, chosen_action: int, stop_prior: float) -> bool:
        if self.steps_in_milestone < self.min_steps_per_milestone:
            return False
        if self.completion < self.advance_completion:
            return False
        return chosen_action == STOP or stop_prior >= self.stop_prior_threshold


def _action_increment(action_id: int) -> float:
    if action_id in {MOVE_FORWARD, GO_UP, GO_DOWN, MOVE_LEFT, MOVE_RIGHT}:
        return 0.08
    if action_id in {TURN_LEFT, TURN_RIGHT}:
        return 0.04
    return 0.0


def _from_global_progress(progress: float, num_milestones: int) -> tuple[int, float]:
    if num_milestones <= 0:
        return 0, float(progress)
    index = max(0, min(num_milestones - 1, int(progress * num_milestones)))
    local_start = index / num_milestones
    local_end = (index + 1) / num_milestones
    completion = (progress - local_start) / max(1e-6, local_end - local_start)
    return index, float(np.clip(completion, 0.0, 1.0))


def _milestones(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
    milestones = item.get("milestones") or item.get("instruction_plan") or []
    if isinstance(milestones, dict):
        milestones = milestones.get("milestones") or []
    return list(milestones)


def _milestone_text(milestone: Mapping[str, Any]) -> str:
    return (
        f"{milestone.get('action_type', '')} {milestone.get('spatial_relation', '')} "
        f"{', '.join(milestone.get('landmarks', []))}. "
        f"cues: {'; '.join(milestone.get('verification_cues', []))}"
    ).strip()
