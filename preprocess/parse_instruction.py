from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.llm_clients import build_llm_client
from preprocess.common import instruction_id, instruction_text, load_episodes, write_jsonl
from schemas.milestone_schema import validate_instruction_plan
from utils.diagnostics import append_jsonl, ensure_dir, print_event, save_bar_png, write_json


def build_user_prompt(item_id: str, text: str) -> str:
    return (
        f"instruction_id: {item_id}\n"
        f"instruction: {text}\n\n"
        "Return JSON matching the schema in the system prompt. "
        "Use 3 to 8 milestones and include concrete verification cues."
    )


def parse_dataset(args: argparse.Namespace) -> None:
    diag_dir = ensure_dir(args.diagnostics_dir)
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    episodes = load_episodes(args.dataset_json)
    client = build_llm_client(args.client)
    good_rows = []
    bad_rows = []
    milestone_hist: Dict[str, float] = {}
    print_event("parse_instruction", "start", dataset=args.dataset_json, client=args.client, episodes=len(episodes))
    for episode in episodes:
        item_id = instruction_id(episode)
        text = instruction_text(episode)
        last_error = ""
        for _ in range(args.max_retries + 1):
            try:
                raw = client.generate_json(system_prompt, build_user_prompt(item_id, text))
                raw["instruction_id"] = str(raw.get("instruction_id") or item_id)
                plan = validate_instruction_plan(raw)
                good_rows.append(plan)
                milestone_hist[str(len(plan["milestones"]))] = milestone_hist.get(str(len(plan["milestones"])), 0.0) + 1.0
                print_event("parse_instruction", "parsed", instruction_id=item_id, milestones=len(plan["milestones"]))
                append_jsonl(diag_dir / "events.jsonl", {"stage": "parse_instruction", "instruction_id": item_id, "milestones": len(plan["milestones"])})
                last_error = ""
                break
            except Exception as exc:
                last_error = str(exc)
        if last_error:
            bad_rows.append({"instruction_id": item_id, "instruction": text, "error": last_error})
            print_event("parse_instruction", "bad_case", instruction_id=item_id, error=last_error)
    write_jsonl(args.output, good_rows)
    if bad_rows:
        write_jsonl(args.bad_output, bad_rows)
    write_json(diag_dir / "summary.json", {"total": len(episodes), "valid": len(good_rows), "bad": len(bad_rows)})
    save_bar_png(diag_dir / "milestone_count_distribution.png", "Milestone count distribution", milestone_hist)
    print_event("parse_instruction", "done", valid=len(good_rows), bad=len(bad_rows), diagnostics=str(diag_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--output", default="data/instruction_plan.jsonl")
    parser.add_argument("--bad-output", default="data/bad_cases.jsonl")
    parser.add_argument("--prompt", default="prompts/milestone_prompt.txt")
    parser.add_argument("--client", choices=["rule", "qwen_api", "qwen_local"], default="rule")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--diagnostics-dir", default="DATA/v0/diagnostics/parse_instruction")
    parse_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
