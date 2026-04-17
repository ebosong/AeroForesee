from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.diagnostics import print_event, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="configs/fuser.yaml")
    parser.add_argument("--w-progress", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--w-cost", nargs="+", type=float, default=[0.3, 0.7, 1.0])
    parser.add_argument("--w-prior", nargs="+", type=float, default=[0.0, 0.3, 0.6])
    args = parser.parse_args()
    # Placeholder calibration: writes the first grid point. Replace score_fn with val-unseen eval once data is ready.
    best = next(itertools.product(args.w_progress, args.w_cost, args.w_prior))
    data = {
        "w_progress": best[0],
        "w_cost": best[1],
        "w_prior": best[2],
        "fallback": {
            "low_progress_threshold": 0.05,
            "low_score_threshold": 0.05,
            "progress_patience": 3,
            "repeated_turn_patience": 2,
            "conservative_actions": ["MOVE_FORWARD", "STOP"],
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    write_json(Path(args.output).with_suffix(".summary.json"), data)
    print_event("tune_fuser", "wrote_config", output=args.output, w_progress=best[0], w_cost=best[1], w_prior=best[2])


if __name__ == "__main__":
    main()
