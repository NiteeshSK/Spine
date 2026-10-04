"""Config loading and run provenance.

Every run dumps ``run_meta.json`` next to its adapter: config, git SHA, package
versions, GPU name. Six weeks later, "which run produced the 0.94?" has an
answer.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """``--set train.learning_rate=1e-4 lora.r=32`` style overrides."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node[parts[-1]] = value
    return cfg


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def run_metadata(cfg: dict) -> dict:
    meta: dict[str, Any] = {
        "config": cfg,
        "git_sha": _sh(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            meta["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        meta["torch"] = "not installed"
    for pkg in ("transformers", "trl", "peft", "datasets", "bitsandbytes", "accelerate"):
        try:
            mod = __import__(pkg)
            meta[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            meta[pkg] = "not installed"
    return meta


def save_metadata(cfg: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_meta.json"
    path.write_text(json.dumps(run_metadata(cfg), indent=2, default=str))
    return path
