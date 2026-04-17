from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from models.action_space import AirVLNActionSpace


@dataclass
class FallbackState:
    progress_history: deque = field(default_factory=lambda: deque(maxlen=3))
    action_history: deque = field(default_factory=lambda: deque(maxlen=2))


class FallbackPolicy:
    def __init__(
        self,
        action_space: AirVLNActionSpace | None = None,
        low_progress_threshold: float = 0.05,
        low_score_threshold: float = 0.05,
        progress_patience: int = 3,
        repeated_turn_patience: int = 2,
        conservative_actions: Iterable[str] | None = None,
    ) -> None:
        self.action_space = action_space or AirVLNActionSpace()
        self.low_progress_threshold = low_progress_threshold
        self.low_score_threshold = low_score_threshold
        self.progress_patience = max(1, int(progress_patience))
        self.repeated_turn_patience = max(1, int(repeated_turn_patience))
        self.conservative_actions = list(conservative_actions or ["MOVE_FORWARD", "STOP"])
        self.state = FallbackState(
            progress_history=deque(maxlen=self.progress_patience),
            action_history=deque(maxlen=self.repeated_turn_patience),
        )

    def select(self, ranked_scores: Dict[int, float], progress_gain: float) -> Tuple[int, bool]:
        if not ranked_scores:
            return self.action_space.id_from_official_name("STOP"), True
        best_action = max(ranked_scores, key=ranked_scores.get)
        trigger = self._should_trigger(ranked_scores, progress_gain)
        if trigger:
            best_action = self._conservative_choice(ranked_scores)
        self.state.progress_history.append(float(progress_gain))
        self.state.action_history.append(int(best_action))
        return int(best_action), trigger

    def _should_trigger(self, scores: Dict[int, float], progress_gain: float) -> bool:
        recent_low = list(self.state.progress_history) + [float(progress_gain)]
        if len(recent_low) >= self.progress_patience and all(v < self.low_progress_threshold for v in recent_low[-self.progress_patience:]):
            return True
        turn_left = self.action_space.id_from_official_name("TURN_LEFT")
        turn_right = self.action_space.id_from_official_name("TURN_RIGHT")
        turn_ids = {turn_left, turn_right}
        recent_actions = list(self.state.action_history) + [max(scores, key=scores.get)]
        if len(recent_actions) >= self.repeated_turn_patience and all(action in turn_ids for action in recent_actions[-self.repeated_turn_patience:]):
            return True
        return max(scores.values()) < self.low_score_threshold

    def _conservative_choice(self, scores: Dict[int, float]) -> int:
        stop = self.action_space.id_from_official_name("STOP")
        top2 = [action for action, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:2]]
        conservative_ids: List[int] = []
        for name in self.conservative_actions:
            try:
                conservative_ids.append(self.action_space.id_from_official_name(str(name)))
            except KeyError:
                continue
        for action in conservative_ids:
            if action in top2:
                return action
        available = [action for action in conservative_ids if action in scores]
        if available:
            return max(available, key=lambda action: scores[action])
        return top2[-1] if top2 else stop

