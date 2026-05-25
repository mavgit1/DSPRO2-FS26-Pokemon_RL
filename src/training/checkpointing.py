import os
import shutil
from pathlib import Path
from typing import List
import threading # <--- ADD THIS

class CheckpointManager:
    """
    Manages model checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_num: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_num = keep_num
        self.checkpoints: List[Path] = []

    def save_checkpoint(self, algo, step: int) -> Path:
        """Save checkpoint and manage rotation."""
        save_result = algo.save(str(self.checkpoint_dir / f"step_{step}"))

        real_path = save_result.checkpoint.path
        checkpoint_path = Path(real_path)
        self.checkpoints.append(checkpoint_path)

        while len(self.checkpoints) > self.keep_num:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                shutil.rmtree(old_ckpt, ignore_errors=True)

        if os.environ.get("SAVE_LEAGUE_HISTORY") == "1":
            latest_pt = self.checkpoint_dir / "selfplay_latest.pt"
            history_dir = self.checkpoint_dir / "history"
            
            try:
                if latest_pt.exists():
                    history_dir.mkdir(exist_ok=True)
                    history_file = history_dir / f"selfplay_step_{step}.pt"
                    shutil.copy2(latest_pt, history_file)
            except Exception as e:
                print(f"Archiver failed: {e}")

        return checkpoint_path

    def load_latest(self, algo) -> bool:
        """Load latest checkpoint if available."""
        if not self.checkpoints:
            return False

        latest = self.checkpoints[-1]
        algo.load(str(latest))
        print(f"Loaded checkpoint: {latest}")
        return True