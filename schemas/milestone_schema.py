from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


# 这些动作短句在原始指令中很常见，但往往没有显式 landmark / relation
# 为了避免 parser 因 schema 过严而丢掉样本，这里做一个最小修复：
# 若 milestone 只有动作语义，没有 landmarks 和 spatial_relation，
# 则自动补一个弱 relation="toward"。
ACTION_ONLY_TYPES = {
    "follow",
    "turn",
    "turn_left",
    "turn_right",
    "fly",
    "fly_forward",
    "forward",
    "move",
    "move_forward",
    "ascend",
    "descend",
    "go_up",
    "go_down",
    "takeoff",
    "lift_off",
    "land",
    "stop",
    "approach",
    "fly_over",
}


@dataclass(frozen=True)
class Milestone:
    mid: int
    action_type: str
    landmarks: List[str]
    spatial_relation: str
    verification_cues: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Milestone":
        landmarks = data.get("landmarks") or []
        cues = data.get("verification_cues") or []
        if isinstance(landmarks, str):
            landmarks = [landmarks]
        if isinstance(cues, str):
            cues = [cues]

        action_type = str(data.get("action_type", "")).strip()
        spatial_relation = str(data.get("spatial_relation", "")).strip()
        landmarks = [str(item).strip() for item in landmarks if str(item).strip()]
        cues = [str(item).strip() for item in cues if str(item).strip()]

        # ---- 最小修复逻辑 ----
        # 对于纯动作 milestone（如 fly up / turn left / land），
        # 如果既没有 landmark 也没有 relation，则补一个弱 relation，
        # 让样本可以通过 schema，但又不影响后续字段结构。
        if not landmarks and not spatial_relation:
            normalized_action = action_type.lower().replace(" ", "_")
            if normalized_action in ACTION_ONLY_TYPES:
                spatial_relation = "toward"

        milestone = cls(
            mid=int(data["mid"]),
            action_type=action_type,
            landmarks=landmarks,
            spatial_relation=spatial_relation,
            verification_cues=cues,
        )
        milestone.validate()
        return milestone

    def validate(self) -> None:
        if self.mid < 1:
            raise ValueError("mid must start from 1")
        if not self.action_type:
            raise ValueError("action_type is required")
        if not self.landmarks and not self.spatial_relation:
            raise ValueError("each milestone needs at least one landmark or relation")
        if len(self.verification_cues) < 1:
            raise ValueError("verification_cues needs at least one item")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mid": self.mid,
            "action_type": self.action_type,
            "landmarks": self.landmarks,
            "spatial_relation": self.spatial_relation,
            "verification_cues": self.verification_cues,
        }


@dataclass(frozen=True)
class InstructionPlan:
    instruction_id: str
    milestones: List[Milestone]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionPlan":
        milestones = [Milestone.from_dict(item) for item in data.get("milestones", [])]
        plan = cls(
            instruction_id=str(data.get("instruction_id", "")).strip(),
            milestones=milestones,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.instruction_id:
            raise ValueError("instruction_id is required")
        if not 3 <= len(self.milestones) <= 8:
            raise ValueError("milestones length must be in [3, 8]")
        mids = [item.mid for item in self.milestones]
        expected = list(range(1, len(mids) + 1))
        if mids != expected:
            raise ValueError(f"mid must be consecutive: expected {expected}, got {mids}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "milestones": [item.to_dict() for item in self.milestones],
        }


def validate_instruction_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    return InstructionPlan.from_dict(data).to_dict()