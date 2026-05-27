#!/usr/bin/env python3
"""
Train AlphaZero (Track A).

Supports both TicTacToe (for core verification) and chess.
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "chess"],
                        help="Game to train on")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=100,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Handicap pattern (chess only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detect if not given)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Create game
    if args.game == "tictactoe":
        from handichess.track_a.game.tictactoe import TicTacToeGame
        game = TicTacToeGame()
        default_sims = 50
        mcts_config = {"dirichlet_alpha": 1.0, "temperature_threshold": 6}
        net_config = {"num_res_blocks": 4, "num_channels": 64}
    else:
        from handichess.track_a.game.chess_std import ChessGame
        if args.pattern:
            game = ChessGame.from_handicap(args.pattern)
        else:
            game = ChessGame()
        default_sims = 800
        mcts_config = {"dirichlet_alpha": 0.3, "temperature_threshold": 30}
        net_config = {"num_res_blocks": 10, "num_channels": 128}

    num_sims = args.simulations or default_sims
    mcts_config["num_simulations"] = num_sims

    # Create net
    from handichess.track_a.net import create_net_for_game
    net = create_net_for_game(game, net_config)
    logger.info(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Create trainer
    from handichess.track_a.trainer import Trainer
    trainer = Trainer(
        net, device=device,
        checkpoint_dir=args.checkpoint_dir,
        config={"epochs_per_iteration": 10, "batch_size": 64},
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create self-play
    from handichess.track_a.selfplay import SelfPlay, ReplayBuffer
    selfplay = SelfPlay(
        game, net,
        mcts_config=mcts_config,
        device=device,
    )
    buffer = ReplayBuffer(max_size=50_000)

    # Training loop
    for iteration in range(trainer.iteration, args.iterations):
        logger.info(f"\n{'='*50}")
        logger.info(f"Iteration {iteration + 1}/{args.iterations}")
        logger.info(f"{'='*50}")

        # Self-play
        logger.info(f"Self-play: {args.games_per_iter} games...")
        examples = selfplay.generate_games(args.games_per_iter)
        buffer.add(examples)
        logger.info(f"  Generated {len(examples)} examples (buffer: {len(buffer)})")

        # Train
        train_data = buffer.sample(min(len(buffer), 2048))
        stats = trainer.train(train_data, iteration=iteration)

        # Checkpoint
        if (iteration + 1) % 10 == 0:
            trainer.save_checkpoint()

    # Final save
    trainer.save_checkpoint("final.pt")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
