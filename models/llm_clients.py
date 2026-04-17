from __future__ import annotations

import abc
import json
import re
import subprocess
from typing import Any, Dict, Optional

import requests

from models.qwen_config import (
    QWEN_API_BASE_URL,
    QWEN_API_KEY,
    QWEN_LOCAL_LLM_COMMAND,
    QWEN_TEXT_MODEL,
)


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class QwenAPILLMClient(LLMClient):
    def __init__(self, timeout: int = 120) -> None:
        self.api_key = QWEN_API_KEY
        self.base_url = QWEN_API_BASE_URL.rstrip("/")
        self.model = QWEN_TEXT_MODEL
        self.timeout = timeout

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.api_key or self.api_key == "PASTE_YOUR_QWEN_API_KEY_HERE":
            raise RuntimeError("Set QWEN_API_KEY in models/qwen_config.py before using QwenAPILLMClient")
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return _loads_json_object(text)


class LocalCommandLLMClient(LLMClient):
    def __init__(self, command: Optional[str] = None) -> None:
        self.command = command or QWEN_LOCAL_LLM_COMMAND

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = json.dumps({"system": system_prompt, "user": user_prompt}, ensure_ascii=False)
        completed = subprocess.run(
            self.command,
            input=payload,
            text=True,
            capture_output=True,
            shell=True,
            check=True,
        )
        return _loads_json_object(completed.stdout)


class RuleBasedMilestoneClient(LLMClient):
    """Offline fallback that makes schema-valid milestones without model calls."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        instruction_id = _extract_field(user_prompt, "instruction_id") or "unknown"
        instruction = _extract_field(user_prompt, "instruction") or user_prompt
        parts = [p.strip(" .") for p in re.split(r"[.;]|\bthen\b|\band finally\b", instruction, flags=re.I) if p.strip()]
        if len(parts) < 3:
            words = instruction.split()
            chunk = max(1, len(words) // 3)
            parts = [" ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)]
        parts = parts[:8]
        while len(parts) < 3:
            parts.append(parts[-1] if parts else instruction)
        milestones = []
        for idx, part in enumerate(parts, start=1):
            landmarks = _guess_landmarks(part)
            milestones.append({
                "mid": idx,
                "action_type": _guess_action_type(part),
                "landmarks": landmarks or ["visible route"],
                "spatial_relation": _guess_relation(part),
                "verification_cues": [f"visual cue matches: {landmarks[0] if landmarks else 'route'}"],
            })
        return {"instruction_id": instruction_id, "milestones": milestones}


def build_llm_client(kind: str, **kwargs: Any) -> LLMClient:
    if kind == "qwen_api":
        return QwenAPILLMClient()
    if kind == "qwen_local":
        return LocalCommandLLMClient()
    if kind == "rule":
        return RuleBasedMilestoneClient()
    raise ValueError(f"unknown LLM client kind: {kind}")


def _loads_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("model output does not contain a JSON object")
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model output must be a JSON object")
    return data


def _extract_field(text: str, field: str) -> str:
    match = re.search(rf"{re.escape(field)}\s*:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _guess_action_type(text: str) -> str:
    lower = text.lower()
    for key in ["turn", "descend", "ascend", "land", "follow", "cross", "approach"]:
        if key in lower:
            return key
    if "left" in lower or "right" in lower:
        return "turn"
    if "over" in lower or "through" in lower:
        return "fly_over"
    return "follow"


def _guess_relation(text: str) -> str:
    lower = text.lower()
    for relation in ["along", "through", "over", "near", "left", "right", "toward", "front"]:
        if relation in lower:
            return relation
    return "toward"


def _guess_landmarks(text: str) -> list[str]:
    candidates = [
        "road", "bridge", "building", "tower", "park", "intersection", "river",
        "sign", "billboard", "roof", "street", "square", "shop",
    ]
    lower = text.lower()
    return [item for item in candidates if item in lower][:3]
