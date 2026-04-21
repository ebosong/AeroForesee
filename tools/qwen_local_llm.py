from __future__ import annotations

import json
import sys

import requests


# =========================
# Hard-coded configuration
# =========================
SERVER_URL = "http://127.0.0.1:8002/generate_json"
TIMEOUT = 300

# True: 服务异常时直接失败退出
# False: 尝试做一个最小安全 fallback
STRICT = True


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        _fail(f"Failed to read stdin JSON: {exc}")

    system = payload.get("system")
    user = payload.get("user")

    if not isinstance(system, str) or not isinstance(user, str):
        _fail("Input JSON must contain string fields 'system' and 'user'.")

    try:
        response = requests.post(
            SERVER_URL,
            json={
                "system": system,
                "user": user,
                "max_new_tokens": 1024,
            },
            timeout=TIMEOUT,
        )
    except Exception as exc:
        _fail(f"Request to local LLM server failed: {exc}")

    if response.status_code != 200:
        _fail(f"Local LLM server returned {response.status_code}: {response.text[:2000]}")

    try:
        data = response.json()
    except Exception as exc:
        _fail(f"Failed to parse local server JSON: {exc}. Raw={response.text[:2000]}")

    obj = data.get("json")
    if not isinstance(obj, dict):
        _fail(f"Local server response does not contain a valid 'json' object: {data}")

    # stdout 必须只打印一个 JSON object，供 AeroForesee 的 LocalCommandLLMClient 解析
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _fail(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    if STRICT:
        raise SystemExit(1)

    # 最小 fallback：给一个可过 schema 的 3-step plan
    fallback = {
        "instruction_id": "unknown",
        "milestones": [
            {
                "mid": 1,
                "action_type": "follow",
                "landmarks": ["route"],
                "spatial_relation": "toward",
                "verification_cues": ["progress along route"],
            },
            {
                "mid": 2,
                "action_type": "follow",
                "landmarks": ["target region"],
                "spatial_relation": "toward",
                "verification_cues": ["target region becomes visible"],
            },
            {
                "mid": 3,
                "action_type": "stop",
                "landmarks": ["goal area"],
                "spatial_relation": "near",
                "verification_cues": ["goal area reached"],
            },
        ],
    }
    sys.stdout.write(json.dumps(fallback, ensure_ascii=False))
    sys.stdout.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()