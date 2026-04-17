from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


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
        milestone = cls(
            mid=int(data["mid"]),
            action_type=str(data.get("action_type", "")).strip(),
            landmarks=[str(item).strip() for item in landmarks if str(item).strip()],
            spatial_relation=str(data.get("spatial_relation", "")).strip(),
            verification_cues=[str(item).strip() for item in cues if str(item).strip()],
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

