#!/usr/bin/env python3
"""Run arena evaluation between two checkpoints or against a baseline."""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run arena evaluation")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "chess"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--baseline", type=str, default="random",
                        choices=["random", "greedy", "weak_mcts"],
                        help="Baseline type")
    parser.add_argument("--games", type=int, default=40,
                        help="Number of games")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Handicap pattern (chess only)")
    parser.add_argument("--noq-color", type=str, default="white",
                        choices=["white", "black"],
                        help="Color assigned to the no-queen side for handicap chess")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-simulations", type=int, default=200,
                        help="MCTS simulations per trained-agent move")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSONL summary output path")
    args = parser.parse_args()

    # Create game
    if args.game == "tictactoe":
        from handichess.track_a.game.tictactoe import TicTacToeGame
        game = TicTacToeGame()
        net_config = {"num_res_blocks": 4, "num_channels": 64}
    else:
        from handichess.track_a.game.chess_std import ChessGame
        if args.pattern:
            game = ChessGame.from_matchup(args.pattern, noq_color=args.noq_color)
        else:
            game = ChessGame()
        net_config = {"num_res_blocks": 10, "num_channels": 128}

    # Load net
    from handichess.track_a.net import create_net_for_game
    net = create_net_for_game(game, net_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(args.device)
    net.eval()

    # Evaluate
    from handichess.track_a.baseline import play_against_baseline
    torch.manual_seed(args.seed)
    results = play_against_baseline(
        game, net,
        baseline_type=args.baseline,
        num_games=args.games,
        mcts_config={"num_simulations": args.num_simulations},
        device=args.device,
    )

    print(f"\nResults vs {args.baseline}:")
    print(f"  Wins: {results['wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Losses: {results['losses']}")
    print(f"  Win Rate: {results['win_rate']:.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "game": args.game,
            "checkpoint": args.checkpoint,
            "baseline": args.baseline,
            "games": args.games,
            "pattern": args.pattern,
            "noq_color": args.noq_color if args.game == "chess" else None,
            "q_color": (
                "black" if args.noq_color == "white" else "white"
            ) if args.game == "chess" and args.pattern else None,
            "device": args.device,
            "num_simulations": args.num_simulations,
            "seed": args.seed,
            **results,
        }
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
