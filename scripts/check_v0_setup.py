from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=None, help="Directory that contains AeroForesee, DATA, ENVs, and optional AirVLN.")
    parser.add_argument("--project-dir", default=None, help="AeroForesee project directory. Defaults to this script's repository root.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--fuser-config", default="configs/fuser.yaml")
    parser.add_argument("--eval-split", default="val_unseen")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any warning is found.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve() if args.project_dir else Path(__file__).resolve().parents[1]
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else project_dir.parent.resolve()
    data_dir = workspace_root / "DATA"
    envs_dir = workspace_root / "ENVs"
    failures: list[str] = []
    warnings: list[str] = []

    _check(project_dir.exists(), "project_dir exists", str(project_dir), failures)
    _check(data_dir.exists(), "DATA exists beside AeroForesee", str(data_dir), failures)
    _check(envs_dir.exists(), "ENVs exists beside AeroForesee", str(envs_dir), failures)

    for rel in [
        "data/aerialvln/train.json",
        "data/aerialvln/val_seen.json",
        f"data/aerialvln/{args.eval_split}.json",
        "data/aerialvln-s/train.json",
    ]:
        path = data_dir / rel
        ok = path.exists()
        _check(ok, f"DATA/{rel}", str(path), failures)
        if ok and path.suffix == ".json":
            _preview_json(path, warnings)

    env_count = len(list(envs_dir.glob("env_*"))) if envs_dir.exists() else 0
    _check(env_count > 0, "ENVs/env_* scenes", f"found {env_count}", failures)

    model_cfg_path = project_dir / args.model_config
    fuser_cfg_path = project_dir / args.fuser_config
    _check(model_cfg_path.exists(), "model config", str(model_cfg_path), failures)
    _check(fuser_cfg_path.exists(), "fuser config", str(fuser_cfg_path), failures)
    if model_cfg_path.exists():
        cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8")) or {}
        vision = cfg.get("vision", {})
        for key in ["backbone", "pretrained", "freeze", "dinov2_repo", "torch_hub_dir", "resnet_weights"]:
            _check(key in vision, f"vision.{key} configured", str(vision.get(key)), failures)
        _check_path_config(project_dir, vision.get("dinov2_repo"), "DINOv2 repo path", warnings)
        _check_path_config(project_dir, vision.get("torch_hub_dir"), "Torch Hub cache path", warnings, create_ok=True)
        _check_path_config(project_dir, vision.get("resnet_weights"), "ResNet weights path", warnings, optional=True)

    qwen_config = project_dir / "models" / "qwen_config.py"
    if qwen_config.exists():
        text = qwen_config.read_text(encoding="utf-8")
        if "PASTE_YOUR_QWEN_API_KEY_HERE" in text:
            warnings.append("QWEN_API_KEY is still placeholder. Use --client rule/uniform for smoke tests, or set the key for qwen_api.")

    if _port_open("127.0.0.1", args.port):
        warnings.append(f"Port {args.port} is already open. This is fine if AirVLN simulator server is running; otherwise choose another port.")
    else:
        warnings.append(f"Port {args.port} is not open yet. Start airsim_plugin/AirVLNSimulatorServerTool.py before full eval.")

    print("\n[V0 setup check]")
    print(f"project_dir={project_dir}")
    print(f"workspace_root={workspace_root}")
    for warning in warnings:
        print(f"[WARN] {warning}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print("[OK] Required layout and configs are present.")
    if args.strict and warnings:
        raise SystemExit(2)


def _check(ok: bool, name: str, detail: str, failures: list[str]) -> None:
    print(f"[{'OK' if ok else 'MISS'}] {name}: {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def _preview_json(path: Path, warnings: list[str]) -> None:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
        episodes = data.get("episodes", data) if isinstance(data, dict) else data
        if not isinstance(episodes, list) or not episodes:
            warnings.append(f"{path} does not look like a non-empty episode list.")
    except Exception as exc:
        warnings.append(f"Could not parse {path}: {exc}")


def _check_path_config(project_dir: Path, value: Any, name: str, warnings: list[str], optional: bool = False, create_ok: bool = False) -> None:
    if not value:
        if not optional:
            warnings.append(f"{name} is empty; torch/torchvision defaults will be used.")
        return
    path = Path(str(value))
    if not path.is_absolute():
        path = project_dir / path
    if create_ok:
        path.mkdir(parents=True, exist_ok=True)
    if optional and not path.exists():
        return
    if not path.exists():
        warnings.append(f"{name} does not exist yet: {path}")


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, int(port))) == 0


if __name__ == "__main__":
    main()
