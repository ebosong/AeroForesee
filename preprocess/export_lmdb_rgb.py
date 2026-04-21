from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocess.common import load_episodes, write_jsonl
from utils.diagnostics import ensure_dir, print_event, write_json


KEY_RE = re.compile(r"^(?P<trajectory_id>.+)_(?P<step>\d+)_rgb$")


def export(args: argparse.Namespace) -> None:
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'lmdb'. Install requirements.txt before exporting RGB LMDB.") from exc

    lmdb_dir = Path(args.lmdb_dir)
    output_root = Path(args.output_root)
    diag_dir = ensure_dir(args.diagnostics_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    episode_lookup = _episode_lookup(args.dataset_json)

    rows = []
    exported = 0
    skipped = 0
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
    print_event("export_lmdb_rgb", "start", lmdb_dir=str(lmdb_dir), output_root=str(output_root))
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for raw_key, raw_value in cursor:
            key = raw_key.decode("utf-8")
            parsed = _parse_key(key)
            if parsed is None:
                skipped += 1
                continue
            image = _decode_image(raw_value)
            if image is None:
                skipped += 1
                continue
            trajectory_id = parsed["trajectory_id"]
            step = int(parsed["step"])
            image_path = output_root / _safe_name(trajectory_id) / f"{step:06d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            episode = episode_lookup.get(str(trajectory_id), {})
            rows.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "episode_id": str(episode.get("episode_id", "")) or None,
                    "instruction_id": str(episode.get("episode_id", "")) or str(trajectory_id),
                    "scene_id": episode.get("scene_id"),
                    "step": step,
                    "rgb_path": _portable_path(image_path),
                    "lmdb_key": key,
                }
            )
            exported += 1
            if args.limit > 0 and exported >= args.limit:
                break
    env.close()
    rows.sort(key=lambda row: (str(row["trajectory_id"]), int(row["step"])))
    write_jsonl(args.index_output, rows)
    write_json(
        diag_dir / "summary.json",
        {
            "lmdb_dir": str(lmdb_dir),
            "output_root": str(output_root),
            "index_output": args.index_output,
            "exported": exported,
            "skipped": skipped,
            "dataset_json": args.dataset_json,
        },
    )
    print_event("export_lmdb_rgb", "done", exported=exported, skipped=skipped, index=args.index_output)


def _episode_lookup(dataset_json: str | None) -> Dict[str, Dict[str, Any]]:
    if not dataset_json:
        return {}
    path = Path(dataset_json)
    if not path.exists():
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for episode in load_episodes(path):
        trajectory_id = str(episode.get("trajectory_id", ""))
        episode_id = str(episode.get("episode_id", ""))
        if trajectory_id:
            lookup[trajectory_id] = episode
        if episode_id:
            lookup.setdefault(episode_id, episode)
    return lookup


def _parse_key(key: str) -> Dict[str, str] | None:
    match = KEY_RE.match(key)
    if not match:
        return None
    return match.groupdict()


def _decode_image(raw_value: bytes) -> Any | None:
    try:
        import msgpack_numpy
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for LMDB RGB export. Install msgpack-numpy and Pillow from requirements.txt.") from exc

    try:
        array = msgpack_numpy.unpackb(raw_value, raw=False)
    except Exception:
        return None
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-dir", required=True, help="RGB LMDB directory produced by inference/collect_tf_rgb.py.")
    parser.add_argument("--output-root", default="data/runtime_rgb/train")
    parser.add_argument("--index-output", default="data/runtime_rgb/train_index.jsonl")
    parser.add_argument("--dataset-json", default=None, help="Optional annotation JSON used to add episode_id/scene_id to the index.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/export_lmdb_rgb")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
