from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocess.common import read_jsonl, write_jsonl
from utils.diagnostics import ensure_dir, print_event, write_json


def build(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    windows = list(read_jsonl(args.step_windows))
    print_event("build_latent_targets", "start", windows=len(windows), token_dim=args.token_dim)
    for idx, row in enumerate(windows):
        target = torch.zeros(args.token_dim, dtype=torch.float32)
        target_path = output_dir / f"{row['sample_id']}.pt"
        torch.save(target, target_path)
        index_rows.append({"sample_id": row["sample_id"], "latent_target": str(target_path)})
        if idx < args.preview_count:
            print_event("build_latent_targets", "preview", sample_id=row["sample_id"], target=str(target_path))
    write_jsonl(args.index_output, index_rows)
    write_json(diag_dir / "summary.json", {"step_windows": len(windows), "latent_targets": len(index_rows), "token_dim": args.token_dim})
    print_event("build_latent_targets", "done", latent_targets=len(index_rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--output-dir", default="data/latent_targets/train")
    parser.add_argument("--index-output", default="data/latent_targets/train_index.jsonl")
    parser.add_argument("--token-dim", type=int, default=512)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/build_latent_targets")
    parser.add_argument("--preview-count", type=int, default=10)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
