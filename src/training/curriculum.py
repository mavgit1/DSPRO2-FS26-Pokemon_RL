from collections import deque
from typing import Any, Dict, List, Optional

from src.config.TM_optimal_config import CurriculumConfig, CurriculumStageConfig


class CurriculumManager:
    """
    Manages curriculum learning progression from rolling training outcomes.
    """

    def __init__(
        self,
        curriculum_config: CurriculumConfig,
    ):
        self.config = curriculum_config
        self.stages: List[CurriculumStageConfig] = curriculum_config.stages
        self.current_stage_idx = 0
        self.total_episodes = 0
        self.episodes_in_stage = 0
        self.outcome_window = deque(maxlen=self.config.rolling_window_episodes)

    @property
    def current_stage(self) -> CurriculumStageConfig:
        return self.stages[self.current_stage_idx]

    def _rolling_win_rate(self) -> Optional[float]:
        if not self.outcome_window:
            return None
        return float(sum(self.outcome_window) / len(self.outcome_window))

    def _can_promote(self) -> bool:
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        current = self.current_stage
        rolling_win_rate = self._rolling_win_rate()
        if rolling_win_rate is None:
            return False

        min_samples = max(1, current.min_samples_for_promotion)
        return (
            self.episodes_in_stage >= self.config.min_episodes_before_promotion
            and len(self.outcome_window) >= min_samples
            and rolling_win_rate >= current.promote_at_win_rate
        )

    def update(self, outcomes: List[Optional[int]]) -> bool:
        """
        Update curriculum state from terminal outcomes and check stage progression.

        Returns:
            True if stage changed, False otherwise
        """
        if outcomes:
            self.total_episodes += len(outcomes)
            self.episodes_in_stage += len(outcomes)
            for outcome in outcomes:
                if outcome in {0, 1}:
                    self.outcome_window.append(int(outcome))

        if self._can_promote():
            old_stage = self.current_stage.name
            self.current_stage_idx += 1
            self.episodes_in_stage = 0
            self.outcome_window.clear()
            print(f"Curriculum stage advanced: {old_stage} -> {self.current_stage.name}")
            return True

        return False

    def metrics(self) -> Dict[str, Any]:
        return {
            "curriculum_stage_idx": self.current_stage_idx,
            "curriculum_stage_name": self.current_stage.name,
            "curriculum_total_episodes": self.total_episodes,
            "curriculum_episodes_in_stage": self.episodes_in_stage,
            "curriculum_valid_window_samples": len(self.outcome_window),
            "curriculum_rolling_win_rate": self._rolling_win_rate(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage_idx": self.current_stage_idx,
            "current_stage": self.current_stage.name,
            "total_episodes": self.total_episodes,
            "episodes_in_stage": self.episodes_in_stage,
            "valid_window_samples": len(self.outcome_window),
            "rolling_win_rate": self._rolling_win_rate(),
        }
