from __future__ import annotations

import argparse
import random
from pathlib import Path


def reservoir_sample_jsonl(input_path: str, output_path: str, k: int, seed: int) -> None:
    rng = random.Random(seed)
    reservoir: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line

    if len(reservoir) < k:
        print(f"Warning: only found {len(reservoir)} lines, fewer than requested {k}.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in reservoir:
            f.write(line)

    print(f"Saved {len(reservoir)} sampled step windows to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reservoir_sample_jsonl(
        input_path=args.input,
        output_path=args.output,
        k=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()