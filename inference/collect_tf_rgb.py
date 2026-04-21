from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_collect_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--split", default="train", help="Dataset split under DATA/data/aerialvln/{split}.json.")
    parser.add_argument("--max-episodes", type=int, default=-1, help="Optional cap for quick RGB collection smoke tests.")
    parser.add_argument("--flush-every", type=int, default=1, help="Commit LMDB writes every N minibatches.")
    parser.add_argument("--rgb-only", action="store_true", default=True, help="Force --ablate_depth for faster RGB-only TF collection.")
    args, unknown = parser.parse_known_args()
    return args, unknown


def main() -> None:
    collect_args, airvln_args = parse_collect_args()
    forced = ["--run_type", "collect", "--collect_type", "TF"]
    if collect_args.rgb_only and "--ablate_depth" not in airvln_args:
        forced.append("--ablate_depth")
    sys.argv = [sys.argv[0]] + airvln_args + forced

    from src.common.param import args as old_args
    from src.vlnce_src.env import AirVLNENV
    from utils.diagnostics import ensure_dir, print_event, write_json

    diag_dir = ensure_dir(f"DATA/v0/diagnostics/collect_tf_rgb/{collect_args.split}")
    env = AirVLNENV(batch_size=old_args.batchSize, split=collect_args.split, tokenizer=None)
    collected = 0
    minibatches = 0
    rgb_dir = getattr(env, "lmdb_rgb_dir", "")
    depth_dir = getattr(env, "lmdb_depth_dir", "")

    print_event(
        "collect_tf_rgb",
        "start",
        split=collect_args.split,
        batch_size=old_args.batchSize,
        max_episodes=collect_args.max_episodes,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
    )
    try:
        while collect_args.max_episodes < 0 or collected < collect_args.max_episodes:
            env.next_minibatch()
            if env.batch is None:
                break
            outputs = env.reset()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            for _ in range(int(old_args.maxAction) + 1):
                if np.array(dones).all():
                    break
                teacher_actions = [_teacher_action(obs, done) for obs, done in zip(observations, dones)]
                env.makeActions(teacher_actions)
                outputs = env.get_obs()
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            _mark_collected(env, env.batch)
            collected += len(env.batch)
            minibatches += 1
            if minibatches % max(1, collect_args.flush_every) == 0:
                _flush_lmdb(env)
            print_event("collect_tf_rgb", "minibatch_done", minibatches=minibatches, collected=collected)
    finally:
        _flush_lmdb(env)
        try:
            env.simulator_tool.closeScenes()
        except Exception:
            pass
        gc.collect()

    write_json(
        diag_dir / "summary.json",
        {
            "split": collect_args.split,
            "episodes_collected": collected,
            "minibatches": minibatches,
            "rgb_lmdb_dir": rgb_dir,
            "depth_lmdb_dir": depth_dir,
            "name": old_args.name,
            "project_prefix": old_args.project_prefix,
        },
    )
    print_event("collect_tf_rgb", "done", collected=collected, rgb_dir=rgb_dir)


def _teacher_action(obs: dict[str, Any], done: bool) -> int:
    if done:
        return 0
    value = obs.get("teacher_action", [0])
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _mark_collected(env: Any, batch: list[dict[str, Any]]) -> None:
    if not hasattr(env, "lmdb_features_txn"):
        return
    for item in batch:
        env.lmdb_features_txn.put(str(item["episode_id"]).encode(), b"1")


def _flush_lmdb(env: Any) -> None:
    for name in ("features", "rgb", "depth"):
        txn_name = f"lmdb_{name}_txn"
        env_name = f"lmdb_{name}_env"
        if not hasattr(env, txn_name) or not hasattr(env, env_name):
            continue
        txn = getattr(env, txn_name)
        lmdb_env = getattr(env, env_name)
        try:
            txn.commit()
        except Exception:
            continue
        setattr(env, txn_name, lmdb_env.begin(write=True))


if __name__ == "__main__":
    main()
