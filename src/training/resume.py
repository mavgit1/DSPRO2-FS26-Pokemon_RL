import re
from pathlib import Path
from typing import List, Optional


def extract_step_from_checkpoint_path(checkpoint_path: str) -> Optional[int]:
    match = re.search(r"step_(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def _step_checkpoint_dirs(ckpt_root: Path) -> List[Path]:
    """RLlib new API stack saves under ``checkpoints/step_<n>/``."""
    return sorted(
        [
            p
            for p in ckpt_root.iterdir()
            if p.is_dir() and re.fullmatch(r"step_\d+", p.name)
        ],
        key=lambda p: extract_step_from_checkpoint_path(str(p)) or 0,
    )


def resolve_resume_checkpoint(resume_checkpoint: Optional[str], checkpoint_dir: str) -> Optional[str]:
    if not resume_checkpoint:
        return None

    if resume_checkpoint != "latest":
        path = Path(resume_checkpoint).resolve()
        return str(path) if path.exists() else None

    ckpt_root = Path(checkpoint_dir).resolve()
    if not ckpt_root.exists():
        return None

    # Legacy Tune layout: nested ``checkpoint_*`` directories.
    legacy = [
        p for p in ckpt_root.rglob("*")
        if p.is_dir() and p.name.startswith("checkpoint_")
    ]
    if legacy:
        return str(max(legacy, key=lambda p: p.stat().st_mtime))

    step_dirs = _step_checkpoint_dirs(ckpt_root)
    if step_dirs:
        return str(step_dirs[-1])

    return None
