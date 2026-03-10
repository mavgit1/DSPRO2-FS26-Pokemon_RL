#!/usr/bin/env python
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
import logging

# Configure logging todo: replace with mlflow logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
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
        help="Enable debug logging"
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
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import and run
    from src.config.TM_optimal_config import get_config
    from src.training import PokemonTrainer
    
    logger.info("=" * 60)
    logger.info("Pokemon RL Training")
    logger.info("=" * 60)
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Num servers: {args.num_servers}")
    logger.info(f"Start port: {args.start_port}")
    logger.info(f"Use LSTM: {args.use_lstm}")
    if args.timesteps:
        logger.info(f"Override timesteps: {args.timesteps:,}")
    logger.info("=" * 60)

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