from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from airsim_plugin.airsim_settings import AirsimActions


PROMPT_ALIASES: Dict[str, str] = {
    "STOP": "stop",
    "MOVE_FORWARD": "forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "GO_UP": "ascend",
    "GO_DOWN": "descend",
    "MOVE_LEFT": "move_left",
    "MOVE_RIGHT": "move_right",
}


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    official_name: str
    prompt_name: str


class AirVLNActionSpace:
    def __init__(self) -> None:
        known = dict(AirsimActions._known_actions)
        self._id_to_name = {int(v): str(k) for k, v in known.items()}
        self._name_to_id = {str(k): int(v) for k, v in known.items()}
        self._id_to_prompt = {
            action_id: PROMPT_ALIASES.get(name, name.lower())
            for action_id, name in self._id_to_name.items()
        }
        self._prompt_to_id = {v: k for k, v in self._id_to_prompt.items()}

    @property
    def num_actions(self) -> int:
        return len(self._id_to_name)

    @property
    def action_ids(self) -> List[int]:
        return sorted(self._id_to_name)

    def specs(self) -> List[ActionSpec]:
        return [
            ActionSpec(action_id=i, official_name=self.official_name(i), prompt_name=self.prompt_name(i))
            for i in self.action_ids
        ]

    def official_name(self, action_id: int) -> str:
        return self._id_to_name[int(action_id)]

    def prompt_name(self, action_id: int) -> str:
        return self._id_to_prompt[int(action_id)]

    def id_from_official_name(self, name: str) -> int:
        return self._name_to_id[name]

    def id_from_prompt_name(self, name: str) -> int:
        return self._prompt_to_id[name]

    def valid_ids(self, ids: Iterable[int] | None = None) -> List[int]:
        if ids is None:
            return self.action_ids
        return [int(i) for i in ids if int(i) in self._id_to_name]

    def prompt_action_list(self, ids: Iterable[int] | None = None) -> List[str]:
        return [self.prompt_name(i) for i in self.valid_ids(ids)]

