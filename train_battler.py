"""
Pokemon RL Training Entry Point
===============================

Quick start:
    python train.py                           # Use standard config
    python train.py --preset optimal          # Use optimal config for RTX 5090
    python train.py --preset quick            # Quick test run
    python train.py --timesteps 1000000       # Override timesteps
    python train.py --num-servers 4           # Use 4 Showdown servers
    python train.py --use-lstm                # Enable recurrent model

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
        choices=["quick", "standard", "optimal", "large"],
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
        default=1,
        help="Number of Showdown servers (default: 1)"
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

    # Recurrent model
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help=(
            "Enable recurrent LSTM state path. Disabled by default because "
            "the current RLlib connector setup is non-recurrent."
        ),
    )
    
    args = parser.parse_args()
    
    # Set log level / notify about debug
    if args.debug:
        print("[DEBUG] Debug flag passed. (Note: Standard python logging is disabled in favor of MLflow).")
    
    # Initialize MLflow experiment
    mlflow.set_experiment("Pokemon_RL_Battler")
    
    # Import and run
    from src.config.TM_optimal_config import get_config
    # Make sure this points to your refactored trainer file!
    from src.training.trainer import PokemonTrainer
    
    print("=" * 60)
    print("Pokemon RL Training")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Num servers: {args.num_servers}")
    print(f"Start port: {args.start_port}")
    print(f"Use LSTM: {args.use_lstm}")
    if args.timesteps:
        print(f"Override timesteps: {args.timesteps:,}")
    print("=" * 60)

    config = get_config(args.preset)
    if args.timesteps:
        config.total_timesteps = args.timesteps

    # Keep feed-forward by default to avoid RLlib recurrent connector errors.
    config.model.use_lstm = args.use_lstm

    trainer = PokemonTrainer(
        config=config,
        num_servers=args.num_servers,
        start_port=args.start_port,
    )
    trainer.train()


if __name__ == "__main__":
    main()