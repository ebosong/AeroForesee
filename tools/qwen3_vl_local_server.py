from __future__ import annotations

import os

# =========================
# Hard-coded GPU selection
# =========================
# 只暴露物理 GPU 0 给当前进程
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =========================
# Hard-coded configuration
# =========================
MODEL_PATH = "path/to//Qwen3-VL-8B-Instruct"
HOST = "127.0.0.1"
PORT = 8001

# 先默认关掉，确保先跑通
USE_FLASH_ATTN = False

# 4090 上建议先用 bf16；如果后面不稳可改成 torch.float16
MODEL_DTYPE = torch.bfloat16

# 当前进程可见设备中的第 0 张卡
MODEL_DEVICE = "cuda:0"

DEFAULT_MAX_NEW_TOKENS = 256
MAX_NEW_TOKENS_LIMIT = 512


model: Qwen3VLForConditionalGeneration | None = None
processor: AutoProcessor | None = None


class ScoreRequest(BaseModel):
    prompt: str
    actions: List[str]
    images_base64_png: List[str] = []
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


def _load_model() -> None:
    global model, processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this server is hard-coded to use GPU.")

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": MODEL_DTYPE,
        "device_map": {"": MODEL_DEVICE},
    }
    if USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print("[Local VLM] Loading model...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"[Local VLM] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[Local VLM] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"[Local VLM] torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
    print(f"[Local VLM] current_device={torch.cuda.current_device()}", flush=True)
    print(f"[Local VLM] device_name={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[Local VLM] model device={model.device}", flush=True)
    print("[Local VLM] Model and processor loaded.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Qwen3-VL Local Server", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "host": HOST,
        "port": PORT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "model_device": str(model.device) if model is not None else None,
    }


@app.post("/score_actions")
@torch.inference_mode()
def score_actions(req: ScoreRequest) -> Dict[str, Any]:
    if model is None or processor is None:
        raise RuntimeError("Model or processor is not loaded.")

    action_names = [a.strip() for a in req.actions if a.strip()]
    if not action_names:
        return {"scores": {}, "raw_text": ""}

    system_prompt = (
        "You are a UAV-VLN navigation action-prior evaluator.\n"
        "Score every official action with a number in [0,1].\n"
        "Return JSON only.\n"
        "Keys must exactly match the provided action names.\n"
        "Do not output explanations.\n"
    )

    user_text = (
        f"{req.prompt}\n\n"
        f"Official action list: {json.dumps(action_names, ensure_ascii=False)}\n\n"
        "Return exactly one JSON object like:\n"
        '{"forward": 0.51, "turn_left": 0.18, "turn_right": 0.09, "stop": 0.22}'
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_b64 in req.images_base64_png:
        img_b64 = img_b64.strip()
        if img_b64:
            content.append(
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{img_b64}",
                }
            )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 放到模型所在设备
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=min(max(1, int(req.max_new_tokens)), MAX_NEW_TOKENS_LIMIT),
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    scores = _extract_action_scores(raw_text, action_names)
    return {"scores": scores, "raw_text": raw_text}


def _extract_action_scores(raw_text: str, action_names: List[str]) -> Dict[str, float]:
    data = _try_parse_json_object(raw_text)
    if not isinstance(data, dict):
        return _uniform_scores(action_names)

    scores: Dict[str, float] = {}
    for name in action_names:
        value = data.get(name, 0.0)
        try:
            scores[name] = max(0.0, float(value))
        except Exception:
            scores[name] = 0.0

    total = sum(scores.values())
    if total <= 0:
        return _uniform_scores(action_names)

    return {k: v / total for k, v in scores.items()}


def _try_parse_json_object(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _uniform_scores(action_names: List[str]) -> Dict[str, float]:
    value = 1.0 / max(1, len(action_names))
    return {name: value for name in action_names}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)