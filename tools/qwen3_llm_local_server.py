from __future__ import annotations

import os

# =========================
# Hard-coded GPU selection
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Hard-coded configuration
# =========================
MODEL_PATH = "path/to/Qwen3-8B"
HOST = "127.0.0.1"
PORT = 8002

MODEL_DTYPE = torch.bfloat16
MODEL_DEVICE = "cuda:0"

# 为了结构化输出稳定，默认关闭 thinking
ENABLE_THINKING = False

# 这里优先求稳定 JSON，不采样
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0

DEFAULT_MAX_NEW_TOKENS = 1024
MAX_NEW_TOKENS_LIMIT = 2048


model: AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None


class GenerateJsonRequest(BaseModel):
    system: str
    user: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


def _load_model() -> None:
    global model, tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this server is hard-coded to use GPU.")

    print("[Local LLM] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("[Local LLM] Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=MODEL_DTYPE,
        device_map={"": MODEL_DEVICE},
    )

    print(f"[Local LLM] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[Local LLM] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"[Local LLM] torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
    print(f"[Local LLM] current_device={torch.cuda.current_device()}", flush=True)
    print(f"[Local LLM] device_name={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[Local LLM] model device={model.device}", flush=True)
    print("[Local LLM] Model and tokenizer loaded.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Qwen3-8B Local LLM Server", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "host": HOST,
        "port": PORT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_device": str(model.device) if model is not None else None,
        "enable_thinking": ENABLE_THINKING,
    }


@app.post("/generate_json")
@torch.inference_mode()
def generate_json(req: GenerateJsonRequest) -> Dict[str, Any]:
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer is not loaded.")

    messages = [
        {"role": "system", "content": req.system},
        {"role": "user", "content": req.user},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": min(max(1, int(req.max_new_tokens)), MAX_NEW_TOKENS_LIMIT),
        "do_sample": DO_SAMPLE,
    }
    if DO_SAMPLE:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P
        if TOP_K > 0:
            gen_kwargs["top_k"] = TOP_K

    generated_ids = model.generate(**model_inputs, **gen_kwargs)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    parsed = _extract_json_object(raw_text)

    return {
        "json": parsed,
        "raw_text": raw_text,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Model returned empty text.")

    # Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Extract the first {...} block
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    raise ValueError(f"Model output does not contain a valid JSON object. Raw: {text[:2000]}")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)