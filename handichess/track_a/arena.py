"""
Arena for model evaluation.

Plays games between two agents (or against a baseline) to measure
relative strength. Used for:
  - Deciding whether to accept a new model version
  - Evaluating per-pattern performance (Track A → gamelog)
  - Color-balanced evaluation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import chess

from .game.base import Game
from .net import AlphaZeroNet
from .mcts import MCTS
from handichess.common.gamelog import GameRecord, GameLog, result_from_outcome

logger = logging.getLogger(__name__)


class Arena:
    """
    Plays evaluation games between two MCTS agents.

    Usage:
        arena = Arena(game, net1, net2, config)
        results = arena.play_games(num_games=40)
    """

    def __init__(
        self,
        game: Game,
        net1: AlphaZeroNet,
        net2: AlphaZeroNet,
        mcts_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.game = game
        self.net1 = net1
        self.net2 = net2
        self.device = device

        mc = mcts_config or {}
        self.num_simulations = mc.get("num_simulations", 800)
        self.c_puct = mc.get("c_puct", 1.25)

    def play_game(
        self,
        player1_starts: bool = True,
    ) -> tuple[float, int]:
        """
        Play a single game. Player 1 uses net1, Player 2 uses net2.

        Args:
            player1_starts: If True, net1 plays first (player +1).

        Returns:
            (result_for_player1, num_moves)
            result: +1 win, -1 loss, 0 draw (from player1's perspective).
        """
        mcts1 = MCTS(
            self.game, self.net1,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            device=self.device,
        )
        mcts2 = MCTS(
            self.game, self.net2,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            device=self.device,
        )

        state = self.game.get_init_board()
        player = 1  # +1 starts
        move_count = 0

        while True:
            result = self.game.get_game_ended(state, player)
            if result != 0:
                break

            if move_count >= 512:
                result = 1e-4  # Draw by max moves
                break

            # Determine which MCTS to use
            if (player == 1) == player1_starts:
                mcts = mcts1
            else:
                mcts = mcts2

            # Search with temperature=0 (deterministic)
            action_probs = mcts.search(
                state, player,
                temperature=0.0,
                add_noise=False,
            )

            action = np.argmax(action_probs)
            state, player = self.game.get_next_state(state, player, action)
            move_count += 1

        # Convert result to player1's perspective
        if abs(result) < 0.01:
            return 0.0, move_count
        elif player1_starts:
            # result is from `player`'s perspective at game end
            # If player == 1 (player1's turn) and result > 0 → player1 wins
            if player == 1:
                return float(np.sign(result)), move_count
            else:
                return -float(np.sign(result)), move_count
        else:
            if player == 1:
                return -float(np.sign(result)), move_count
            else:
                return float(np.sign(result)), move_count

    def play_games(
        self, num_games: int
    ) -> dict:
        """
        Play multiple games, color-balanced.

        Half the games: net1 starts. Other half: net2 starts.

        Returns:
            Dict with win/draw/loss counts and win rate for player1 (net1).
        """
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0

        for i in range(num_games):
            player1_starts = (i % 2 == 0)  # Alternate
            result, moves = self.play_game(player1_starts)
            total_moves += moves

            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                draws += 1

        total = wins + draws + losses
        stats = {
            "player1_wins": wins,
            "player1_draws": draws,
            "player1_losses": losses,
            "player1_win_rate": (wins + 0.5 * draws) / max(total, 1),
            "total_games": total,
            "avg_moves": total_moves / max(total, 1),
        }

        logger.info(
            f"Arena: {wins}W / {draws}D / {losses}L "
            f"(win rate: {stats['player1_win_rate']:.3f})"
        )

        return stats


def evaluate_handicap_patterns(
    game_class,
    net: AlphaZeroNet,
    pattern_ids: list[str],
    num_games_per_pattern: int = 40,
    mcts_config: Optional[dict] = None,
    device: str = "cpu",
    log_path: Optional[str] = None,
) -> dict:
    """
    Evaluate a trained net across multiple handicap patterns.
    Net plays both sides; results recorded from handicap side's perspective.

    Args:
        game_class: ChessGame class.
        net: Trained network.
        pattern_ids: List of pattern IDs.
        num_games_per_pattern: Games per pattern (split between colors).
        mcts_config: MCTS configuration.
        device: Torch device.
        log_path: Optional path to save game log.

    Returns:
        Dict mapping pattern_id → {wins, draws, losses, score} for handicap side.
    """
    mc = mcts_config or {}
    game_log = GameLog(log_path) if log_path else None
    results = {}

    for pid in pattern_ids:
        wins = draws = losses = 0

        for side in ["white", "black"]:
            game = game_class.from_handicap(pid, side)
            mcts = MCTS(
                game, net,
                num_simulations=mc.get("num_simulations", 400),
                c_puct=mc.get("c_puct", 1.25),
                device=device,
            )

            games_per_side = num_games_per_pattern // 2

            for g in range(games_per_side):
                state = game.get_init_board()
                player = 1
                move_count = 0

                while move_count < 512:
                    result = game.get_game_ended(state, player)
                    if result != 0:
                        break

                    probs = mcts.search(state, player, temperature=0.0, add_noise=False)
                    action = np.argmax(probs)
                    state, player = game.get_next_state(state, player, action)
                    move_count += 1

                # Determine handicap side result
                game_result = game.get_game_ended(state, player)
                if abs(game_result) < 0.01:
                    draws += 1
                    hresult, hscore = "draw", 0.5
                elif (game_result > 0 and player == 1 and side == "white") or \
                     (game_result > 0 and player == -1 and side == "black"):
                    wins += 1
                    hresult, hscore = "win", 1.0
                else:
                    losses += 1
                    hresult, hscore = "loss", 0.0

                if game_log:
                    game_log.write(GameRecord(
                        pattern_id=pid,
                        handicap_side=side,
                        result=hresult,
                        result_score=hscore,
                        ply=move_count,
                        start_fen=game.start_fen,
                        termination="completed",
                        engine="az_trained",
                        nodes=mc.get("num_simulations", 400),
                    ))

        total = wins + draws + losses
        results[pid] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": (wins + 0.5 * draws) / max(total, 1),
            "total": total,
        }

        logger.info(f"Pattern {pid}: {wins}W/{draws}D/{losses}L = {results[pid]['score']:.3f}")

    return results
