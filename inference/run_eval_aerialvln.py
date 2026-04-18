from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.device import select_torch_device


def parse_v0_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--v0-checkpoint")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--fuser-config", default="configs/fuser.yaml")
    parser.add_argument("--vlm-client", choices=["uniform", "qwen_api", "qwen_local"], default="uniform")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated physical GPU ids exposed to the V0 planner process, e.g. 0 or 0,1.")
    parser.add_argument("--eval-output", default="DATA/v0/eval/results.json")
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/eval_aerialvln")
    parser.add_argument("--instruction-plan", default="data/instruction_plan.jsonl")
    parser.add_argument("--stop-completion-threshold", type=float, default=0.35)
    parser.add_argument("--score-preview-steps", type=int, default=5)
    args, unknown = parser.parse_known_args()
    return args, unknown


def main() -> None:
    v0_args, airvln_args = parse_v0_args()
    device_selection = select_torch_device(v0_args.device, v0_args.gpu_ids)
    forced = ["--run_type", "eval", "--ablate_depth"]
    sys.argv = [sys.argv[0]] + airvln_args + forced

    from src.common.param import args as old_args
    from src.vlnce_src.env import AirVLNENV
    from models.action_prior import ActionPriorModule
    from models.action_space import AirVLNActionSpace
    from models.causal_latent_action_evaluator import CausalLatentActionEvaluator
    from models.decision_fuser import DecisionFuser
    from models.state_builder import MilestoneAwareStateBuilder
    from models.vlm_clients import build_vlm_client
    from inference.planner_loop import V0PlannerLoop
    from preprocess.common import read_jsonl
    from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_png, write_json

    old_args.ablate_depth = True
    diag_dir = ensure_dir(v0_args.diagnostics_dir)
    device = device_selection.device
    cfg = yaml.safe_load(open(v0_args.model_config, "r", encoding="utf-8"))
    token_dim = int(cfg["model"]["hidden_dim"])
    action_space = AirVLNActionSpace()
    state_builder = MilestoneAwareStateBuilder(
        token_dim=token_dim,
        action_space=action_space,
        vision_backbone=str(cfg["vision"].get("backbone", "dinov2_s")),
        vision_pretrained=bool(cfg["vision"].get("pretrained", False)),
        vision_freeze=bool(cfg["vision"].get("freeze", True)),
        dinov2_repo=str(cfg["vision"].get("dinov2_repo") or "") or None,
        torch_hub_dir=str(cfg["vision"].get("torch_hub_dir") or "") or None,
        resnet_weights=str(cfg["vision"].get("resnet_weights") or "") or None,
    )
    evaluator = CausalLatentActionEvaluator(
        num_actions=action_space.num_actions,
        token_dim=token_dim,
        num_heads=int(cfg["model"]["num_heads"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    if v0_args.v0_checkpoint:
        ckpt = torch.load(v0_args.v0_checkpoint, map_location="cpu")
        state_builder.load_state_dict(ckpt["state_builder"], strict=False)
        evaluator.load_state_dict(ckpt["evaluator"], strict=False)

    vlm_client = build_vlm_client(v0_args.vlm_client)
    fuser = DecisionFuser.from_yaml(v0_args.fuser_config)
    planner = V0PlannerLoop(
        state_builder=state_builder,
        evaluator=evaluator,
        action_prior=ActionPriorModule(
            action_space=action_space,
            vlm_client=vlm_client,
            stop_completion_threshold=v0_args.stop_completion_threshold,
        ),
        fuser=fuser,
        device=device,
        history_len=int(cfg["trajectory"]["history_len"]),
        max_keyframes=int(cfg["history"]["max_keyframes"]),
        fallback_config=fuser.fallback_config,
        score_preview_steps=v0_args.score_preview_steps,
        diagnostics_dir=str(diag_dir / "planner_loop"),
    )

    env = AirVLNENV(batch_size=old_args.batchSize, split=old_args.EVAL_DATASET, tokenizer=None)
    plans_loaded, plans_merged = _merge_instruction_plans(env.data, v0_args.instruction_plan, read_jsonl)
    if plans_loaded:
        print_event("run_eval_aerialvln", "instruction_plans_merged", loaded=plans_loaded, merged=plans_merged, path=v0_args.instruction_plan)
    stats_episodes = {}
    print_event(
        "run_eval_aerialvln",
        "start",
        split=old_args.EVAL_DATASET,
        batch_size=old_args.batchSize,
        ablate_depth=old_args.ablate_depth,
        device=str(device),
        gpu_ids=",".join(device_selection.gpu_ids) or "all",
        cuda_visible_devices=device_selection.cuda_visible_devices or "",
        device_note=device_selection.note,
    )
    try:
        for _ in range(0, len(env.data), env.batch_size):
            env.next_minibatch()
            if env.batch is None:
                break
            outputs = env.reset()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            for step in range(int(old_args.maxAction)):
                result = planner.act(observations, env.batch, step)
                append_jsonl(diag_dir / "eval_steps.jsonl", {"batch_episode_ids": [item["episode_id"] for item in env.batch], "step": step, "actions": result["actions"], "fallbacks": result["fallbacks"]})
                env.makeActions(result["actions"])
                outputs = env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                if np.array(dones).all():
                    break
            for idx, item in enumerate(env.batch):
                stats_episodes[str(item["episode_id"])] = _to_jsonable(infos[idx])
                print_event("run_eval_aerialvln", "episode_done", episode_id=item["episode_id"], success=infos[idx].get("success"), ndtw=infos[idx].get("ndtw"), steps=infos[idx].get("steps_taken"))
            if old_args.EVAL_NUM != -1 and len(stats_episodes) >= old_args.EVAL_NUM:
                break
    finally:
        try:
            env.simulator_tool.closeScenes()
        except Exception:
            pass
        gc.collect()

    output = Path(v0_args.eval_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = _aggregate(stats_episodes)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": stats_episodes}, f, ensure_ascii=False, indent=2)
    metric_values = {key: value for key, value in summary.items() if key != "num_episodes" and isinstance(value, (int, float))}
    if metric_values:
        save_bar_png(diag_dir / "eval_metrics.png", "Eval mean metrics", metric_values)
    write_json(
        diag_dir / "summary.json",
        {
            "episodes": len(stats_episodes),
            "output": str(output),
            "summary": summary,
            "instruction_plans_loaded": plans_loaded,
            "instruction_plans_merged": plans_merged,
            "device": str(device),
            "gpu_ids": device_selection.gpu_ids,
            "cuda_visible_devices": device_selection.cuda_visible_devices,
            "device_note": device_selection.note,
        },
    )
    print_event("run_eval_aerialvln", "done", output=str(output), diagnostics=str(diag_dir))


def _aggregate(stats_episodes: dict) -> dict:
    numeric = {}
    for episode in stats_episodes.values():
        for key, value in episode.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    summary = {key: sum(values) / max(1, len(values)) for key, values in numeric.items()}
    summary["num_episodes"] = len(stats_episodes)
    return summary


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _merge_instruction_plans(data: list[dict], path: str, read_jsonl_fn: Any) -> tuple[int, int]:
    plan_path = Path(path)
    if not path or not plan_path.exists():
        return 0, 0
    plans = {str(row.get("instruction_id")): row for row in read_jsonl_fn(plan_path)}
    merged = 0
    for item in data:
        keys = [str(item.get("episode_id", "")), str(item.get("trajectory_id", ""))]
        plan = next((plans[key] for key in keys if key in plans), None)
        if not plan:
            continue
        item["instruction_plan"] = plan
        item["milestones"] = plan.get("milestones", [])
        merged += 1
    return len(plans), merged


if __name__ == "__main__":
    main()
