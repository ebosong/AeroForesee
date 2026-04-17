from __future__ import annotations

import abc
import base64
import json
import subprocess
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from models.qwen_config import (
    QWEN_API_BASE_URL,
    QWEN_API_KEY,
    QWEN_LOCAL_VLM_COMMAND,
    QWEN_VLM_MODEL,
)


class VLMClient(abc.ABC):
    @abc.abstractmethod
    def score_actions(
        self,
        images: List[Image.Image],
        prompt: str,
        action_names: List[str],
    ) -> Dict[str, float]:
        raise NotImplementedError


class QwenAPIVLMClient(VLMClient):
    def __init__(self, timeout: int = 120) -> None:
        self.api_key = QWEN_API_KEY
        self.base_url = QWEN_API_BASE_URL.rstrip("/")
        self.model = QWEN_VLM_MODEL
        self.timeout = timeout

    def score_actions(self, images: List[Image.Image], prompt: str, action_names: List[str]) -> Dict[str, float]:
        if not self.api_key or self.api_key == "PASTE_YOUR_QWEN_API_KEY_HERE":
            raise RuntimeError("Set QWEN_API_KEY in models/qwen_config.py before using QwenAPIVLMClient")
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"},
            })
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        data = json.loads(text)
        return {name: float(data.get(name, 0.0)) for name in action_names}


class LocalCommandVLMClient(VLMClient):
    def __init__(self, command: Optional[str] = None) -> None:
        self.command = command or QWEN_LOCAL_VLM_COMMAND

    def score_actions(self, images: List[Image.Image], prompt: str, action_names: List[str]) -> Dict[str, float]:
        payload = {
            "prompt": prompt,
            "actions": action_names,
            "images_base64_png": [_encode_image(image) for image in images],
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            shell=True,
            check=True,
        )
        data = json.loads(completed.stdout)
        return {name: float(data.get(name, 0.0)) for name in action_names}


class UniformVLMClient(VLMClient):
    def score_actions(self, images: List[Image.Image], prompt: str, action_names: List[str]) -> Dict[str, float]:
        value = 1.0 / max(1, len(action_names))
        return {name: value for name in action_names}


def build_vlm_client(kind: str, **kwargs: Any) -> VLMClient:
    if kind == "qwen_api":
        return QwenAPIVLMClient()
    if kind == "qwen_local":
        return LocalCommandVLMClient()
    if kind == "uniform":
        return UniformVLMClient()
    raise ValueError(f"unknown VLM client kind: {kind}")


def _encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
