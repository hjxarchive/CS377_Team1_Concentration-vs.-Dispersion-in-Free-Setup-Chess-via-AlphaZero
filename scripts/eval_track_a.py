#!/usr/bin/env python3
"""Run Track A evaluation using the robust evaluate_matchup_patterns pipeline."""

import argparse
import logging
import torch

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.arena import evaluate_matchup_patterns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run Track A Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=400,
                        help="Number of games per pattern (total)")
    parser.add_argument("--pattern", type=str, required=True,
                        help="Handicap pattern id")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-simulations", type=int, default=400,
                        help="MCTS simulations per move")
    parser.add_argument("--output", type=str, required=True,
                        help="JSONL summary output path")
    args = parser.parse_args()

    game = ChessGame()
    net_config = {"num_res_blocks": 10, "num_channels": 128}

    # Load net
    net = create_net_for_game(game, net_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(args.device)
    net.eval()

    # The function expects num_games_per_pattern to be the number of games per side (white/black).
    # Since args.games is the total, we divide by 2.
    games_per_side = max(1, args.games // 2)

    evaluate_matchup_patterns(
        game_class=ChessGame,
        net=net,
        pattern_ids=[args.pattern],
        num_games_per_pattern=games_per_side,
        mcts_config={"num_simulations": args.num_simulations},
        device=args.device,
        log_path=args.output
    )

if __name__ == "__main__":
    main()
