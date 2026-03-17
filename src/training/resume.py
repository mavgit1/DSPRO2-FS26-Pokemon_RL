import re
from pathlib import Path
from typing import Optional


def extract_step_from_checkpoint_path(checkpoint_path: str) -> Optional[int]:
    match = re.search(r"step_(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def resolve_resume_checkpoint(resume_checkpoint: Optional[str], checkpoint_dir: str) -> Optional[str]:
    if not resume_checkpoint:
        return None

    if resume_checkpoint != "latest":
        return resume_checkpoint

    ckpt_root = Path(checkpoint_dir).resolve()
    if not ckpt_root.exists():
        return None

    candidates = [
        p for p in ckpt_root.rglob("*")
        if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)
