from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.models.vocab import vocab_sizes

_VOCAB_SIZES = vocab_sizes()


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Embedding dimensions
    num_tokens: int = 13
    token_dim: int = 164
    species_vocab_size: int = _VOCAB_SIZES["species_vocab_size"]
    item_vocab_size: int = _VOCAB_SIZES["item_vocab_size"]
    ability_vocab_size: int = _VOCAB_SIZES["ability_vocab_size"]
    embedding_dim: int = 32

    # Transformer
    hidden_dim: int = 256
    num_heads: int = 4
    num_transformer_layers: int = 1
    dropout: float = 0.05
    use_position_embeddings: bool = True
    use_role_embeddings: bool = True

    # LSTM (for memory across turns)
    lstm_hidden: int = 512
    use_lstm: bool = True
    max_seq_len: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "token_dim": self.token_dim,
            "species_vocab_size": self.species_vocab_size,
            "item_vocab_size": self.item_vocab_size,
            "ability_vocab_size": self.ability_vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_transformer_layers": self.num_transformer_layers,
            "dropout": self.dropout,
            "use_position_embeddings": self.use_position_embeddings,
            "use_role_embeddings": self.use_role_embeddings,
            "lstm_hidden": self.lstm_hidden,
            "use_lstm": self.use_lstm,
            "max_seq_len": self.max_seq_len,
        }


@dataclass
class PPOConfig:
    """Standard PPO hyperparameters."""

    # Learning
    lr: float = 5e-4

    # Discount and GAE
    gamma: float = 0.97
    lambda_: float = 0.92

    # PPO clipping
    clip_param: float = 0.15

    # Entropy bonus (exploration)
    entropy_coeff: float = 0.05

    # Value function
    vf_loss_coeff: float = 0.5
    vf_clip_param: float = 10.0

    # Gradient clipping
    grad_clip: float = 5.0

    # Batch sizes
    train_batch_size: int = 4096
    sgd_minibatch_size: int = 512
    num_sgd_iter: int = 8          


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    # Battle settings
    battle_format: str = "gen8randombattlenogimmicks"

    # Fixed player team (Showdown format text file). When set, the RL agent
    # always uses this team and the battle format is auto-switched to the
    # corresponding custom-game variant (e.g. gen5randombattle → gen5customgame).
    player_team_path: Optional[str] = "data/teams/player_team.txt"

    # Server settings
    showdown_host: str = "localhost"
    start_port: int = 8000
    num_servers: int = 8

    # Parallelism
    # Number of RLlib rollout workers (processes) collecting experience.
    # Scale up until CPU cores or Showdown servers are saturated; increasing this
    # usually improves sample throughput but also increases RAM usage.
    num_workers: int = 24
    # Number of battle environments run concurrently inside each worker.
    # Effective parallel envs ~= num_workers * num_envs_per_worker. Increase this
    # when workers are underutilized; reduce it if memory pressure or env lag appears.
    num_envs_per_worker: int = 8


@dataclass
class RewardConfig:
    """Reward function configuration."""

    # Major events
    victory_reward: float = 10.0
    defeat_penalty: float = -10.0

    # HP-based rewards
    hp_value_weight: float = 2.0

    # Fainting rewards
    fainted_value: float = 5.0
    fainted_penalty: float = -5.0

    # Progress rewards
    step_penalty: float = -0.005

    # Type matchup shaping: gentle nudge toward favorable matchups (0.2 ≪ terminal ±10).
    matchup_reward_weight: float = 0.2

    # Action quality: per-step signal for picking effective moves (0.3 ≪ terminal ±10).
    action_quality_weight: float = 0.3


@dataclass
class CurriculumStageConfig:
    """Single curriculum stage settings.

    ``opponent_mix`` weights use opponent keys consumed by ``battle_env``:
    ``random`` (poke-env ``RandomPlayer``), ``random_no_switch`` (random among
    moves only; no voluntary switches), ``heuristic``, ``self``. See README /
    CLAUDE.md (2026-05-04).
    """

    name: str
    promote_at_win_rate: float
    min_samples_for_promotion: int = 50
    opponent_mix: Dict[str, float] = field(default_factory=lambda: {"random": 1.0})
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "promote_at_win_rate": self.promote_at_win_rate,
            "min_samples_for_promotion": self.min_samples_for_promotion,
            "opponent_mix": dict(self.opponent_mix),
            "reward_config": {
                "victory_reward": self.reward_config.victory_reward,
                "defeat_penalty": self.reward_config.defeat_penalty,
                "hp_value_weight": self.reward_config.hp_value_weight,
                "fainted_value": self.reward_config.fainted_value,
                "fainted_penalty": self.reward_config.fainted_penalty,
                "step_penalty": self.reward_config.step_penalty,
                "matchup_reward_weight": self.reward_config.matchup_reward_weight,
                "action_quality_weight": self.reward_config.action_quality_weight,
            },
        }


@dataclass
class CurriculumConfig:
    """Curriculum configuration with per-stage payloads."""

    enabled: bool = True
    rolling_window_episodes: int = 300
    min_episodes_before_promotion: int = 3_000
    allow_demotion: bool = False
    reward_rollback_on_demotion: bool = False
    stages: List[CurriculumStageConfig] = field(
        default_factory=lambda: [
            CurriculumStageConfig(
                name="moves_only_warmup",
                promote_at_win_rate=0.65,
                min_samples_for_promotion=200,
                opponent_mix={"random": 1.0},
                reward_config=RewardConfig(
                    victory_reward=8.0,
                    defeat_penalty=-10.0,
                    hp_value_weight=3.0,      
                    fainted_value=6.0,
                    fainted_penalty=-6.0,
                    step_penalty=-0.01,        
                    matchup_reward_weight=0.1, 
                    action_quality_weight=0.3,
                ),
            ),
            CurriculumStageConfig(
                name="random_with_switches",
                promote_at_win_rate=0.72,
                min_samples_for_promotion=200,
                opponent_mix={"random_no_switch": 0.6, "random": 0.4},
                reward_config=RewardConfig(
                    victory_reward=10.0,
                    defeat_penalty=-10.0,
                    hp_value_weight=3.0,      
                    fainted_value=6.0,
                    fainted_penalty=-6.0,
                    step_penalty=-0.01,        
                    matchup_reward_weight=0.1, 
                    action_quality_weight=0.3,
                ),
            ),
            CurriculumStageConfig(
                name="self_play",
                promote_at_win_rate=0.8,
                min_samples_for_promotion=300,
                opponent_mix={"self": 0.7, "random": 0.2, "random_no_switch": 0.1},
                reward_config=RewardConfig(
                    victory_reward=10.0,
                    defeat_penalty=-10.0,
                    hp_value_weight=3.0,      
                    fainted_value=6.0,
                    fainted_penalty=-6.0,
                    step_penalty=-0.01,        
                    matchup_reward_weight=0.1, 
                    action_quality_weight=0.4,
                ),
            ),
            CurriculumStageConfig(
                name="mixed_final",
                promote_at_win_rate=1.01,
                min_samples_for_promotion=300,
                opponent_mix={"heuristic": 0.5, "self": 0.3, "random_no_switch": 0.2},
                reward_config=RewardConfig(
                    victory_reward=10.0,
                    defeat_penalty=-10.0,
                    hp_value_weight=3.0,      
                    fainted_value=6.0,
                    fainted_penalty=-6.0,
                    step_penalty=-0.01,        
                    matchup_reward_weight=0.1, 
                    action_quality_weight=0.4,
                ),
            ),
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rolling_window_episodes": self.rolling_window_episodes,
            "min_episodes_before_promotion": self.min_episodes_before_promotion,
            "allow_demotion": self.allow_demotion,
            "reward_rollback_on_demotion": self.reward_rollback_on_demotion,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass
class ValidationScheduleConfig:
    """Scheduled checkpoint validation during training."""

    enabled: bool = True
    freq_steps: int = 100_000
    protocols: List[str] = field(
        default_factory=lambda: ["smoke", "fixed_paired", "mirror"]
    )
    fixed_pair_manifest: str = "data/validation/gen8_random_battle_team_pairs.json"
    mirror_manifest: str = "data/validation/gen8_random_battle_mirror_teams.json"
    max_steps_per_battle: int = 500
    seed: int = 42
    num_servers: int = 1
    continue_on_failure: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "freq_steps": self.freq_steps,
            "protocols": list(self.protocols),
            "fixed_pair_manifest": self.fixed_pair_manifest,
            "mirror_manifest": self.mirror_manifest,
            "max_steps_per_battle": self.max_steps_per_battle,
            "seed": self.seed,
            "num_servers": self.num_servers,
            "continue_on_failure": self.continue_on_failure,
        }


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    # Duration
    total_timesteps: int = 3_000_000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 150_000      # Save every N timesteps
    keep_checkpoints_num: int = 5
    
    # Logging
    log_dir: str = "logs"
    print_freq: int = 100_000           # Print every N timesteps
    
    # Hardware
    num_gpus: float = 1.0
    num_gpus_per_worker: float = 0.0
    
    # Curriculum
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    
    # Evaluation
    evaluation_interval: int = 100_000
    evaluation_duration: int = 100
    validation: ValidationScheduleConfig = field(default_factory=ValidationScheduleConfig)
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Self-play
    selfplay_weights_path: str = "checkpoints/selfplay_latest.pt"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_timesteps": self.total_timesteps,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_freq": self.checkpoint_freq,
            "num_gpus": self.num_gpus,
            "selfplay_weights_path": self.selfplay_weights_path,
            "curriculum": self.curriculum.to_dict(),
            "validation": self.validation.to_dict(),
            "model": self.model.to_dict(),
            "ppo": {
                "lr": self.ppo.lr,
                "gamma": self.ppo.gamma,
                "lambda_": self.ppo.lambda_,
                "clip_param": self.ppo.clip_param,
                "entropy_coeff": self.ppo.entropy_coeff,
                "vf_loss_coeff": self.ppo.vf_loss_coeff,
                "train_batch_size": self.ppo.train_batch_size,
                "sgd_minibatch_size": self.ppo.sgd_minibatch_size,
                "num_sgd_iter": self.ppo.num_sgd_iter,
            },
            "env": {
                "battle_format": self.env.battle_format,
                "player_team_path": self.env.player_team_path,
                "num_workers": self.env.num_workers,
                "num_envs_per_worker": self.env.num_envs_per_worker,
            },
            "reward": {
                "victory_reward": self.reward.victory_reward,
                "defeat_penalty": self.reward.defeat_penalty,
                "hp_value_weight": self.reward.hp_value_weight,
                "fainted_value": self.reward.fainted_value,
                "fainted_penalty": self.reward.fainted_penalty,
                "step_penalty": self.reward.step_penalty,
                "matchup_reward_weight": self.reward.matchup_reward_weight,
                "action_quality_weight": self.reward.action_quality_weight,
            },
        }


# =============================================================================
# PRESETS
# =============================================================================

def get_config(preset: str = "standard") -> TrainingConfig:
    """Get a configuration preset."""
    
    presets = {
        "quick": TrainingConfig(
            total_timesteps=150_000,
            env=EnvironmentConfig(
                num_workers=12,
                num_envs_per_worker=4,
            ),
            model=ModelConfig(
                hidden_dim=128,
                num_transformer_layers=1,
                use_lstm=False,
            ),
            ppo=PPOConfig(
                train_batch_size=4096,
            ),
        ),
        
        "standard": TrainingConfig(
            # Uses all defaults
        ),

        "memory_safe": TrainingConfig(
            env=EnvironmentConfig(
                num_workers=4,
                num_envs_per_worker=2,
            ),
            model=ModelConfig(
                hidden_dim=256,
                num_heads=4,
                num_transformer_layers=2,
                lstm_hidden=256,
                use_lstm=False,
            ),
            ppo=PPOConfig(
                train_batch_size=2048,
                sgd_minibatch_size=128,
            ),
        ),
        
        "large": TrainingConfig(
            env=EnvironmentConfig(
                num_workers=12,
                num_envs_per_worker=4,
            ),
            model=ModelConfig(
                hidden_dim=768,
                num_heads=12,
                num_transformer_layers=6,
                lstm_hidden=768,
            ),
            ppo=PPOConfig(
                train_batch_size=16384,
                sgd_minibatch_size=512,
            ),
        ),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]