from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.vision_backbone import VisionBackbone
from preprocess.common import read_jsonl, write_jsonl
from utils.diagnostics import ensure_dir, print_event, write_json


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    windows = list(read_jsonl(args.step_windows))
    cfg = yaml.safe_load(open(args.model_config, "r", encoding="utf-8"))
    token_dim = int(args.token_dim or cfg["model"]["hidden_dim"])
    image_h = int(cfg["vision"]["image_height"])
    image_w = int(cfg["vision"]["image_width"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    encoder = VisionBackbone(
        token_dim=token_dim,
        backbone=str(cfg["vision"].get("backbone", "dinov2_s")),
        pretrained=bool(cfg["vision"].get("pretrained", True)),
        freeze=True,
        dinov2_repo=str(cfg["vision"].get("dinov2_repo") or "") or None,
        torch_hub_dir=str(cfg["vision"].get("torch_hub_dir") or "") or None,
        resnet_weights=str(cfg["vision"].get("resnet_weights") or "") or None,
    ).to(device).eval()
    encoded = 0
    missing = 0
    image_root = Path(args.image_root) if args.image_root else None
    print_event("build_latent_targets", "start", windows=len(windows), token_dim=token_dim, device=device)
    for idx, row in enumerate(windows):
        image = _load_rgb(row.get("next_rgb_path"), image_h, image_w, image_root)
        if image is None:
            target = torch.zeros(token_dim, dtype=torch.float32)
            missing += 1
        else:
            with torch.no_grad():
                target = encoder(image.to(device))[0].squeeze(0).detach().cpu().float()
            encoded += 1
        target_path = output_dir / f"{row['sample_id']}.pt"
        torch.save(target, target_path)
        index_rows.append({"sample_id": row["sample_id"], "latent_target": str(target_path)})
        if idx < args.preview_count:
            print_event("build_latent_targets", "preview", sample_id=row["sample_id"], target=str(target_path), encoded=bool(image is not None))
    write_jsonl(args.index_output, index_rows)
    write_json(diag_dir / "summary.json", {"step_windows": len(windows), "latent_targets": len(index_rows), "token_dim": token_dim, "encoded_from_images": encoded, "missing_images_zero_fallback": missing})
    print_event("build_latent_targets", "done", latent_targets=len(index_rows), encoded=encoded, missing_images=missing, diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--output-dir", default="data/latent_targets/train")
    parser.add_argument("--index-output", default="data/latent_targets/train_index.jsonl")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--image-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-dim", type=int)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_latent_targets")
    parser.add_argument("--preview-count", type=int, default=10)
    build(parser.parse_args())


def _resolve_image_path(path: str | None, image_root: Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() and image_root is not None:
        candidate = image_root / candidate
    return candidate if candidate.exists() else None


def _load_rgb(path: str | None, height: int, width: int, image_root: Path | None) -> torch.Tensor | None:
    resolved = _resolve_image_path(path, image_root)
    if resolved is None:
        return None
    try:
        from PIL import Image

        image = Image.open(resolved).convert("RGB").resize((width, height))
        return torch.tensor(np.asarray(image), dtype=torch.float32).unsqueeze(0)
    except Exception:
        return None


if __name__ == "__main__":
    main()
