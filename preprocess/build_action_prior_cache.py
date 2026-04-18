from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.action_prior import ActionPriorModule
from models.action_space import AirVLNActionSpace
from models.vlm_clients import build_vlm_client
from preprocess.common import read_jsonl, write_jsonl
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_png, write_json


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    action_space = AirVLNActionSpace()
    client = build_vlm_client(args.client)
    prior_module = ActionPriorModule(action_space=action_space, vlm_client=client, prompt_path=args.prompt)
    rows = []
    top_action_hist = {action_space.official_name(i): 0.0 for i in action_space.action_ids}
    windows = list(read_jsonl(args.step_windows))
    image_root = Path(args.image_root) if args.image_root else None
    current_rgb_loaded = 0
    keyframes_loaded = 0
    print_event("build_action_prior_cache", "start", windows=len(windows), client=args.client)
    for idx, row in enumerate(windows):
        current_rgb = _load_rgb(row.get("rgb_path"), image_root)
        keyframes = [
            image
            for image in (_load_rgb(path, image_root) for path in (row.get("keyframe_rgb_paths") or [])[-args.max_keyframes:])
            if image is not None
        ]
        current_rgb_loaded += int(current_rgb is not None)
        keyframes_loaded += len(keyframes)
        scores = prior_module.score(
            current_rgb=current_rgb,
            keyframes=keyframes[-2:],
            milestone_text=row["milestone_text"],
            progress_summary=_progress_summary(row),
            legal_action_ids=row.get("legal_action_ids"),
            milestone_completion=float(row.get("completion", 0.0)),
        )
        rows.append({
            "sample_id": row["sample_id"],
            "prior": {str(k): v for k, v in scores.items()},
        })
        top_action = max(scores, key=scores.get)
        top_action_hist[action_space.official_name(top_action)] += 1.0
        if idx < args.preview_count:
            append_jsonl(diag_dir / "prior_preview.jsonl", {"sample_id": row["sample_id"], "top_action": action_space.official_name(top_action), "scores": {action_space.official_name(k): v for k, v in scores.items()}})
            print_event("build_action_prior_cache", "preview", sample_id=row["sample_id"], top_action=action_space.official_name(top_action), top_score=f"{scores[top_action]:.4f}")
    write_jsonl(args.output, rows)
    write_json(
        diag_dir / "summary.json",
        {
            "step_windows": len(windows),
            "cached_priors": len(rows),
            "client": args.client,
            "current_rgb_loaded": current_rgb_loaded,
            "keyframes_loaded": keyframes_loaded,
        },
    )
    save_bar_png(diag_dir / "top_prior_action_distribution.png", "Top prior action distribution", top_action_hist)
    print_event("build_action_prior_cache", "done", cached=len(rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--output", default="data/action_prior_cache/train.jsonl")
    parser.add_argument("--prompt", default="prompts/action_prior_prompt.txt")
    parser.add_argument("--client", choices=["uniform", "qwen_api", "qwen_local"], default="uniform")
    parser.add_argument("--image-root")
    parser.add_argument("--max-keyframes", type=int, default=2)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_action_prior_cache")
    parser.add_argument("--preview-count", type=int, default=20)
    build(parser.parse_args())


def _load_rgb(path: str | None, image_root: Path | None) -> Image.Image | None:
    resolved = _resolve_image_path(path, image_root)
    if resolved is None:
        return None
    try:
        return Image.open(resolved).convert("RGB")
    except Exception:
        return None


def _resolve_image_path(path: str | None, image_root: Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() and image_root is not None:
        candidate = image_root / candidate
    return candidate if candidate.exists() else None


def _progress_summary(row: dict) -> str:
    fields: List[str] = [f"milestone_completion={float(row.get('completion', 0.0)):.3f}"]
    if "global_completion" in row:
        fields.append(f"global_completion={float(row.get('global_completion', 0.0)):.3f}")
    if row.get("action_history"):
        fields.append(f"recent_actions={row['action_history'][-4:]}")
    return "; ".join(fields)


if __name__ == "__main__":
    main()
