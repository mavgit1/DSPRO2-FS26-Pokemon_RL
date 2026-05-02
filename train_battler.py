"""
Pokemon RL Training Entry Point
===============================

Quick start:
    python train.py                           # Use standard config
    python train.py --preset optimal          # Use optimal config for RTX 5090
    python train.py --preset memory_safe      # Lower RAM pressure setup
    python train.py --preset quick            # Quick test run
    python train.py --timesteps 1000000       # Override timesteps
    python train.py --num-servers 4           # Use 4 Showdown servers

Requirements:
    - Pokemon Showdown server(s) running
    - All dependencies installed (see requirements.txt)
"""

import argparse
import mlflow
from dotenv import load_dotenv, find_dotenv

def main():
    load_dotenv(find_dotenv())
    
    parser = argparse.ArgumentParser(
        description="Train Pokemon RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        choices=["quick", "standard", "memory_safe", "optimal", "large"],
        help="Configuration preset (default: standard)"
    )
    
    # Timesteps
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
    )
    
    # Servers
    parser.add_argument(
        "--num-servers",
        type=int,
        default=8,
        help="Number of Showdown servers (default: 8, must match running servers)"
    )
    
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="Starting port for Showdown servers (default: 8000)"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (Note: standard logging replaced by MLflow/prints)"
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to RLlib checkpoint directory to restore from. "
            "Use 'latest' to auto-pick the most recent checkpoint in checkpoint_dir."
        ),
    )

    parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="MLflow run ID to continue logging into the same run.",
    )

    parser.add_argument(
        "--disable-scheduled-validation",
        action="store_true",
        help="Disable automatic checkpoint validation during training.",
    )

    parser.add_argument(
        "--validation-freq-steps",
        type=int,
        default=None,
        help="Override scheduled validation frequency in environment steps.",
    )

    parser.add_argument(
        "--validation-max-steps-per-battle",
        type=int,
        default=None,
        help="Override scheduled validation battle truncation length.",
    )
    
    args = parser.parse_args()
    
    # Set log level / notify about debug
    if args.debug:
        print("[DEBUG] Debug flag passed. (Note: Standard python logging is disabled in favor of MLflow).")
    
    # Initialize MLflow experiment
    mlflow.set_experiment("Pokemon_RL_Battler")
    
    # Import and run. Change this to own config file if you want.
    from src.config.TM_optimal_config import get_config

    from src.training.trainer import PokemonTrainer
    
    print("=" * 60)
    print("Pokemon RL Training")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Num servers: {args.num_servers}")
    print(f"Start port: {args.start_port}")
    if args.resume_checkpoint:
        print(f"Resume checkpoint: {args.resume_checkpoint}")
    if args.mlflow_run_id:
        print(f"MLflow run id: {args.mlflow_run_id}")
    if args.timesteps:
        print(f"Override timesteps: {args.timesteps:,}")
    print("=" * 60)

    config = get_config(args.preset)
    if args.timesteps:
        config.total_timesteps = args.timesteps

    if args.disable_scheduled_validation:
        config.validation.enabled = False
    if args.validation_freq_steps is not None:
        config.validation.freq_steps = args.validation_freq_steps
    if args.validation_max_steps_per_battle is not None:
        config.validation.max_steps_per_battle = args.validation_max_steps_per_battle

    trainer = PokemonTrainer(
        config=config,
        preset=args.preset,
        num_servers=args.num_servers,
        start_port=args.start_port,
        resume_checkpoint=args.resume_checkpoint,
        mlflow_run_id=args.mlflow_run_id,
    )
    trainer.train()


if __name__ == "__main__":
    main()