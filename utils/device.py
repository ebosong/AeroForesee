from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    requested_device: str
    gpu_ids: list[str]
    cuda_visible_devices: str | None
    cuda_available: bool
    note: str


def normalize_gpu_ids(gpu_ids: str | Iterable[int | str] | None) -> list[str]:
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, str):
        value = gpu_ids.strip()
        if not value or value.lower() in {"all", "none", "auto"}:
            return []
        parts = value.split(",")
    else:
        parts = list(gpu_ids)
    normalized: list[str] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if not text.isdigit():
            raise ValueError(f"Invalid GPU id '{text}'. Use a comma-separated list like 0,1.")
        normalized.append(text)
    return normalized


def select_torch_device(requested_device: str = "cuda", gpu_ids: str | Iterable[int | str] | None = None) -> DeviceSelection:
    gpu_list = normalize_gpu_ids(gpu_ids)
    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    requested = (requested_device or "cuda").strip().lower()
    if requested == "cpu":
        return DeviceSelection(torch.device("cpu"), requested, gpu_list, visible, torch.cuda.is_available(), "forced_cpu")

    if not requested.startswith("cuda"):
        return DeviceSelection(torch.device(requested), requested, gpu_list, visible, torch.cuda.is_available(), "custom_device")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return DeviceSelection(torch.device("cpu"), requested, gpu_list, visible, False, "cuda_unavailable_fallback_cpu")

    if requested == "cuda" and gpu_list:
        requested = "cuda:0"
    return DeviceSelection(torch.device(requested), requested, gpu_list, visible, True, "cuda")
