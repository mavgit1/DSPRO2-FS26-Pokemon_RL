from dataclasses import dataclass, field
from typing import List, Dict, Any

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
    hidden_dim: int = 512
    num_heads: int = 8
    num_transformer_layers: int = 4
    dropout: float = 0.1
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
    lr: float = 3.0e-4

    # Discount and GAE
    gamma: float = 0.99
    lambda_: float = 0.95

    # PPO clipping
    clip_param: float = 0.2

    # Entropy bonus (exploration)
    entropy_coeff: float = 0.02

    # Value function
    vf_loss_coeff: float = 0.5
    vf_clip_param: float = 5.0

    # Gradient clipping
    grad_clip: float = 0.5

    # Batch sizes
    train_batch_size: int = 6144
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10          


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    # Battle settings
    battle_format: str = "gen8randombattle"

    # Server settings
    showdown_host: str = "localhost"
    start_port: int = 8000
    num_servers: int = 8

    # Parallelism
    num_workers: int = 12
    num_envs_per_worker: int = 4


@dataclass
class RewardConfig:
    """Reward function configuration."""

    # Major events
    victory_reward: float = 100.0
    defeat_penalty: float = -100.0

    # HP-based rewards
    hp_value_weight: float = 1.0

    # Fainting rewards
    fainted_value: float = 3.0
    fainted_penalty: float = -3.0

    # Progress rewards
    step_penalty: float = -0.02


@dataclass
class CurriculumStageConfig:
    """Single curriculum stage settings."""

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
            },
        }


@dataclass
class CurriculumConfig:
    """Curriculum configuration with per-stage payloads."""

    enabled: bool = True
    rolling_window_episodes: int = 200
    min_episodes_before_promotion: int = 100
    allow_demotion: bool = False
    reward_rollback_on_demotion: bool = False
    stages: List[CurriculumStageConfig] = field(
        default_factory=lambda: [
            CurriculumStageConfig(
                name="innit",
                promote_at_win_rate=0.75,
                min_samples_for_promotion=50,
                opponent_mix={"random": 0.8, "heuristic": 0.2},
                reward_config=RewardConfig(
                    victory_reward=80.0,
                    defeat_penalty=-80.0,
                    hp_value_weight=1.2,
                    fainted_value=3.0,
                    fainted_penalty=-2.0,
                    step_penalty=-0.005,
                ),
            ),
            CurriculumStageConfig(
                name="easy",
                promote_at_win_rate=0.75,
                min_samples_for_promotion=50,
                opponent_mix={"random": 0.65, "heuristic": 0.35},
                reward_config=RewardConfig(
                    victory_reward=100.0,
                    defeat_penalty=-100.0,
                    hp_value_weight=1.0,
                    fainted_value=3.0,
                    fainted_penalty=-3.0,
                    step_penalty=-0.01,
                ),
            ),
            CurriculumStageConfig(
                name="medium",
                promote_at_win_rate=0.75,
                min_samples_for_promotion=50,
                opponent_mix={"random": 0.4, "heuristic": 0.6},
                reward_config=RewardConfig(
                    victory_reward=100.0,
                    defeat_penalty=-100.0,
                    hp_value_weight=1.0,
                    fainted_value=3.0,
                    fainted_penalty=-3.0,
                    step_penalty=-0.01,
                ),
            ),
            CurriculumStageConfig(
                name="advanced",
                promote_at_win_rate=0.75,
                min_samples_for_promotion=50,
                opponent_mix={"random": 0.2, "heuristic": 0.8},
                reward_config=RewardConfig(
                    victory_reward=100.0,
                    defeat_penalty=-100.0,
                    hp_value_weight=1.0,
                    fainted_value=3.0,
                    fainted_penalty=-3.0,
                    step_penalty=-0.01,
                ),
            ),
            CurriculumStageConfig(
                name="hard",
                promote_at_win_rate=1.01,
                min_samples_for_promotion=50,
                opponent_mix={"heuristic": 1.0},
                reward_config=RewardConfig(
                    victory_reward=120.0,
                    defeat_penalty=-120.0,
                    hp_value_weight=0.8,
                    fainted_value=4.0,
                    fainted_penalty=-3.0,
                    step_penalty=-0.02,
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
    total_timesteps: int = 10_000_000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 500_000      # Save every N timesteps
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_timesteps": self.total_timesteps,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_freq": self.checkpoint_freq,
            "num_gpus": self.num_gpus,
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
            },
        }


# =============================================================================
# PRESETS
# =============================================================================

def get_config(preset: str = "standard") -> TrainingConfig:
    """Get a configuration preset."""
    
    presets = {
        "quick": TrainingConfig(
            total_timesteps=1_000_000,
            env=EnvironmentConfig(
                num_workers=0,
                num_envs_per_worker=1,
            ),
            model=ModelConfig(
                hidden_dim=128,
                num_transformer_layers=1,
                use_lstm=False,
            ),
            ppo=PPOConfig(
                train_batch_size=2048,
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