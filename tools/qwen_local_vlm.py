from __future__ import annotations

import json
import sys
from typing import Dict, List

import requests


# =========================
# Hard-coded configuration
# =========================
SERVER_URL = "http://127.0.0.1:8001/score_actions"
TIMEOUT = 300

# True: 服务异常时直接报错退出
# False: 服务异常时回退为 uniform 分布，避免整条预处理任务中断
STRICT = False


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        _fail(f"Failed to read stdin JSON: {exc}", [])

    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        _fail("Field 'actions' must be a list.", [])

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=TIMEOUT)
    except Exception as exc:
        _fail(f"Request to local VLM server failed: {exc}", actions)

    if response.status_code != 200:
        _fail(f"Local VLM server returned {response.status_code}: {response.text[:2000]}", actions)

    try:
        data = response.json()
    except Exception as exc:
        _fail(f"Failed to parse local server JSON: {exc}. Raw={response.text[:2000]}", actions)

    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        _fail(f"Local server response does not contain a valid 'scores' object: {data}", actions)

    sys.stdout.write(json.dumps(scores, ensure_ascii=False))
    sys.stdout.flush()


def _fail(message: str, actions: List[str]) -> None:
    print(message, file=sys.stderr, flush=True)

    if STRICT:
        raise SystemExit(1)

    names = [str(x).strip() for x in actions if str(x).strip()]
    if names:
        fallback = _uniform_scores(names)
        sys.stdout.write(json.dumps(fallback, ensure_ascii=False))
        sys.stdout.flush()
        raise SystemExit(0)

    raise SystemExit(1)


def _uniform_scores(action_names: List[str]) -> Dict[str, float]:
    value = 1.0 / max(1, len(action_names))
    return {name: value for name in action_names}


if __name__ == "__main__":
    main()