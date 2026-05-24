import random
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from poke_env.player import RandomPlayer
from poke_env.battle.abstract_battle import AbstractBattle

# Import your SelfPlayPlayer to inherit its inference engine
from src.training.self_play_player import SelfPlayPlayer

class HistoricalSelfPlayer(SelfPlayPlayer):
    """
    Self-play opponent that selects a historical checkpoint 
    and uses it for a batch of battles to simulate league play.
    """
    max_history_snapshots: int = 5

    def __init__(
        self, 
        model_config_dict: Dict[str, Any], 
        weights_path: Optional[str] = None,
        max_history_snapshots: int = 5,
        **kwargs
    ):
        super().__init__(
            model_config_dict=model_config_dict, 
            weights_path=weights_path, 
            **kwargs
        )
        self.max_history_snapshots = max(1, int(max_history_snapshots))
        # MEMORY LEAK FIX 1: Force this specific opponent model strictly to the CPU.
        # This keeps the VRAM entirely free for the main RLlib trainer.
        self.model.to("cpu")
        
        self._current_brain_path = None
        self._battles_played_with_current = 0

    def _try_load_weights(self) -> None:
        # Override the base class's per-turn mtime check.
        pass

    def _load_random_historical_brain(self) -> None:
        """Pick a random brain from the most recent N checkpoint snapshots."""
        if not self._weights_path:
            return
            
        base_dir = Path(self._weights_path).parent
        history_dir = base_dir / "history"
        
        pt_files = []
        current_time = time.time()
        
        if history_dir.exists() and history_dir.is_dir():
            for f in history_dir.glob("selfplay_step_*.pt"):
                try:
                    if current_time - f.stat().st_mtime > 10.0:
                        pt_files.append(f)
                except OSError:
                    pass
            # Fallback: any stale .pt in history/
            if not pt_files:
                for f in history_dir.glob("*.pt"):
                    try:
                        if current_time - f.stat().st_mtime > 10.0:
                            pt_files.append(f)
                    except OSError:
                        pass

        def _step_from_path(path: Path) -> int:
            name = path.stem
            if name.startswith("selfplay_step_"):
                try:
                    return int(name.split("_")[-1])
                except ValueError:
                    return 0
            return int(path.stat().st_mtime)

        if pt_files:
            pt_files.sort(key=_step_from_path)
            pt_files = pt_files[-self.max_history_snapshots :]
                    
        # Fallback to the active weights if history is empty
        if not pt_files:
            target_path = Path(self._weights_path)
            try:
                if target_path.exists() and (current_time - target_path.stat().st_mtime > 10.0):
                    pt_files = [target_path]
            except OSError:
                pass
                
        if not pt_files:
            return 

        target_path = random.choice(pt_files)
        
        if target_path == self._current_brain_path:
            self._battles_played_with_current = 0
            return
            
        try:
            state_dict = torch.load(target_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            
            self._lstm_states.clear()
            self._load_count += 1
            self._diag["weight_load_count"] += 1
            self._current_brain_path = target_path
            self._battles_played_with_current = 0  
            
            # MEMORY LEAK FIX 2: Explicitly sever the reference to the loaded tensors 
            # and force Python's garbage collector to run instantly.
            del state_dict
            gc.collect()
            
        except Exception:
            pass

    def choose_move(self, battle: AbstractBattle):
        # Clean up memory
        if getattr(battle, "finished", False) or battle.won or battle.lost:
            self._lstm_states.pop(battle.battle_tag, None)

        if len(self._lstm_states) > 50:
            oldest_keys = list(self._lstm_states.keys())[:-20]
            for k in oldest_keys:
                self._lstm_states.pop(k, None)

        # If it's a new battle
        if battle.battle_tag not in self._lstm_states:
            self._battles_played_with_current += 1
            
            # ZIP ERROR FIX 2 (I/O Shield): Don't swap brains every single battle. 
            # Play 20 matches with a historical model before swapping. 
            # This drops disk usage by 95% and stops the stuttering.
            if self._current_brain_path is None or self._battles_played_with_current > 20:
                self._load_random_historical_brain()
                
            self._lstm_states[battle.battle_tag] = None

        try:
            return self._inference_move(battle)
        except Exception:
            self._diag["fallback_count"] += 1
            return RandomPlayer.choose_random_singles_move(battle)