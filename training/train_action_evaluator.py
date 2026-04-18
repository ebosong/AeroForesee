from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.action_space import AirVLNActionSpace
from models.causal_latent_action_evaluator import CausalLatentActionEvaluator
from models.state_builder import MilestoneAwareStateBuilder
from training.v0_dataset import V0ActionDataset, collate_v0
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_line_png, write_json


def train(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    model_cfg = yaml.safe_load(open(args.model_config, "r", encoding="utf-8"))
    token_dim = int(model_cfg["model"]["hidden_dim"])
    image_h = int(model_cfg["vision"]["image_height"])
    image_w = int(model_cfg["vision"]["image_width"])
    action_space = AirVLNActionSpace()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    dataset = V0ActionDataset(
        step_windows=args.step_windows,
        rollout_labels=args.rollout_labels,
        action_prior_cache=args.action_prior_cache,
        latent_index=args.latent_index,
        image_height=image_h,
        image_width=image_w,
        max_keyframes=int(model_cfg["history"]["max_keyframes"]),
        image_root=args.image_root,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_v0)
    print_event("train_action_evaluator", "dataset_loaded", samples=len(dataset), batches=len(loader), device=device)

    state_builder = MilestoneAwareStateBuilder(
        token_dim=token_dim,
        action_space=action_space,
        vision_backbone=str(model_cfg["vision"].get("backbone", "dinov2_s")),
        vision_pretrained=bool(model_cfg["vision"].get("pretrained", False)),
        vision_freeze=bool(model_cfg["vision"].get("freeze", True)),
        dinov2_repo=str(model_cfg["vision"].get("dinov2_repo") or "") or None,
        torch_hub_dir=str(model_cfg["vision"].get("torch_hub_dir") or "") or None,
        resnet_weights=str(model_cfg["vision"].get("resnet_weights") or "") or None,
    ).to(device)
    evaluator = CausalLatentActionEvaluator(
        num_actions=action_space.num_actions,
        token_dim=token_dim,
        num_heads=int(model_cfg["model"]["num_heads"]),
        num_layers=int(model_cfg["model"]["num_layers"]),
        dropout=float(model_cfg["model"]["dropout"]),
    ).to(device)

    params = list(state_builder.history_encoder.parameters())
    params += list(state_builder.trajectory_encoder.parameters())
    params += list(state_builder.milestone_id_embedding.parameters())
    params += list(state_builder.text_encoder.parameters())
    params += list(state_builder.progress_mlp.parameters())
    params += list(evaluator.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_curve = []
    for epoch in range(args.epochs):
        state_builder.train()
        evaluator.train()
        running = 0.0
        for batch_idx, batch in enumerate(loader):
            batch = _to_device(batch, device)
            state = state_builder(
                current_rgb=batch["current_rgb"],
                history_rgbs=batch["history_rgbs"],
                action_history=batch["action_history"],
                pose_deltas=batch["pose_deltas"],
                milestone_ids=batch["milestone_id"],
                milestone_texts=batch["milestone_text"],
                completion=batch["completion"],
                recent_progress_flag=batch["recent_progress_flag"],
                prev_latent=batch["prev_latent"],
                fallback_flags=batch["fallback_flags"],
            )
            pred = evaluator(state, batch["action_id"])
            loss_progress = F.mse_loss(pred["progress_gain"], batch["progress_label"])
            loss_cost = F.mse_loss(pred["cost"], batch["cost_label"])
            loss_latent = F.mse_loss(pred["next_latent"], batch["latent_target"])
            loss = (
                args.progress_loss_weight * loss_progress
                + args.cost_loss_weight * loss_cost
                + args.latent_loss_weight * loss_latent
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            if batch_idx < args.preview_batches:
                print_event(
                    "train_action_evaluator",
                    "batch",
                    epoch=epoch,
                    batch=batch_idx,
                    loss=f"{float(loss.detach().cpu()):.6f}",
                    progress_loss=f"{float(loss_progress.detach().cpu()):.6f}",
                    cost_loss=f"{float(loss_cost.detach().cpu()):.6f}",
                    latent_loss=f"{float(loss_latent.detach().cpu()):.6f}",
                )
        denom = max(1, len(loader))
        epoch_loss = running / denom
        loss_curve.append(epoch_loss)
        print_event("train_action_evaluator", "epoch_done", epoch=epoch, loss=f"{epoch_loss:.6f}")
        append_jsonl(diag_dir / "training_log.jsonl", {"epoch": epoch, "loss": epoch_loss})
        save_line_png(diag_dir / "loss_curve.png", "Evaluator training loss", loss_curve)
        torch.save(
            {
                "state_builder": state_builder.state_dict(),
                "evaluator": evaluator.state_dict(),
                "config": model_cfg,
            },
            output_dir / f"ckpt_epoch_{epoch}.pth",
        )
    torch.save(
        {
            "state_builder": state_builder.state_dict(),
            "evaluator": evaluator.state_dict(),
            "config": model_cfg,
        },
        output_dir / "ckpt_last.pth",
    )
    write_json(diag_dir / "summary.json", {"epochs": args.epochs, "samples": len(dataset), "final_loss": loss_curve[-1] if loss_curve else None})
    print_event("train_action_evaluator", "done", checkpoint=str(output_dir / "ckpt_last.pth"), diagnostics=str(diag_dir))


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-windows", required=True)
    parser.add_argument("--rollout-labels", required=True)
    parser.add_argument("--action-prior-cache")
    parser.add_argument("--latent-index")
    parser.add_argument("--image-root")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--output-dir", default="DATA/v0/checkpoints/action_evaluator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--cost-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/train_action_evaluator")
    parser.add_argument("--preview-batches", type=int, default=2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
