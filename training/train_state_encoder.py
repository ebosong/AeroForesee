from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.action_space import AirVLNActionSpace
from models.state_builder import MilestoneAwareStateBuilder
from utils.diagnostics import print_event, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--output", default="DATA/v0/checkpoints/state_builder_init.pth")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.model_config, "r", encoding="utf-8"))
    state_builder = MilestoneAwareStateBuilder(
        token_dim=int(cfg["model"]["hidden_dim"]),
        action_space=AirVLNActionSpace(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_builder": state_builder.state_dict(), "config": cfg}, output)
    write_json(output.parent / "state_builder_init_summary.json", {"checkpoint": str(output), "hidden_dim": int(cfg["model"]["hidden_dim"])})
    print_event("train_state_encoder", "saved_initialization", checkpoint=str(output), hidden_dim=int(cfg["model"]["hidden_dim"]))


if __name__ == "__main__":
    main()
