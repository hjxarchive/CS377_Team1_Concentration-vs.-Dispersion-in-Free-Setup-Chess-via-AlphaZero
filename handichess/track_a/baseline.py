"""
Weak baseline agents for measuring training progress.

Provides random and low-simulation MCTS agents that the trained
AlphaZero should eventually defeat.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .game.base import Game
from .net import AlphaZeroNet
from .mcts import MCTS


class RandomAgent:
    """Plays uniformly random legal moves."""

    def __init__(self, game: Game):
        self.game = game

    def get_action(self, state: np.ndarray, player: int) -> int:
        valid = self.game.get_valid_moves(state, player)
        valid_actions = np.where(valid > 0)[0]
        return np.random.choice(valid_actions)


class GreedyAgent:
    """
    Plays the move with highest immediate value estimate from the net,
    without any search (0 simulations).
    """

    def __init__(
        self,
        game: Game,
        net: AlphaZeroNet,
        device: str = "cpu",
    ):
        self.game = game
        self.net = net
        self.device = device

    def get_action(self, state: np.ndarray, player: int) -> int:
        import torch

        canonical = self.game.get_canonical_form(state, player)
        encoded = self.game.get_encoded_state(canonical)
        valid = self.game.get_valid_moves(state, player)

        encoded_t = torch.FloatTensor(encoded).to(self.device)
        valid_t = torch.FloatTensor(valid).to(self.device)

        policy, _ = self.net.predict(encoded_t, valid_t)
        policy = policy.cpu().numpy()
        policy = policy * valid  # Mask invalid
        return int(np.argmax(policy))


class WeakMCTSAgent:
    """
    MCTS agent with very few simulations — a stepping stone baseline.
    """

    def __init__(
        self,
        game: Game,
        net: AlphaZeroNet,
        num_simulations: int = 25,
        device: str = "cpu",
    ):
        self.game = game
        self.mcts = MCTS(
            game, net,
            num_simulations=num_simulations,
            device=device,
        )

    def get_action(self, state: np.ndarray, player: int) -> int:
        probs = self.mcts.search(state, player, temperature=0.0, add_noise=False)
        return int(np.argmax(probs))


def play_against_baseline(
    game: Game,
    net: AlphaZeroNet,
    baseline_type: str = "random",
    num_games: int = 20,
    mcts_config: Optional[dict] = None,
    device: str = "cpu",
) -> dict:
    """
    Play the trained network against a baseline agent.

    Args:
        game: Game instance.
        net: Trained AlphaZero network.
        baseline_type: "random", "greedy", or "weak_mcts".
        num_games: Number of games (color-balanced).
        mcts_config: MCTS config for the trained net.
        device: Torch device.

    Returns:
        Win rate statistics.
    """
    mc = mcts_config or {}

    # Create baseline
    if baseline_type == "random":
        baseline = RandomAgent(game)
    elif baseline_type == "greedy":
        baseline = GreedyAgent(game, net, device)
    elif baseline_type == "weak_mcts":
        baseline = WeakMCTSAgent(game, net, num_simulations=25, device=device)
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    # Create MCTS for trained agent
    mcts = MCTS(
        game, net,
        num_simulations=mc.get("num_simulations", 200),
        c_puct=mc.get("c_puct", 1.25),
        device=device,
    )

    wins = draws = losses = 0

    for i in range(num_games):
        state = game.get_init_board()
        player = 1
        trained_is_player1 = (i % 2 == 0)
        move_count = 0

        while move_count < 512:
            result = game.get_game_ended(state, player)
            if result != 0:
                break

            is_trained = (player == 1) == trained_is_player1

            if is_trained:
                probs = mcts.search(state, player, temperature=0.0, add_noise=False)
                action = int(np.argmax(probs))
            else:
                action = baseline.get_action(state, player)

            state, player = game.get_next_state(state, player, action)
            move_count += 1

        result = game.get_game_ended(state, player)

        # Convert to trained agent's perspective
        if abs(result) < 0.01:
            draws += 1
        else:
            # Determine if trained agent won
            if trained_is_player1:
                if (player == 1 and result > 0) or (player == -1 and result < 0):
                    wins += 1
                else:
                    losses += 1
            else:
                if (player == -1 and result > 0) or (player == 1 and result < 0):
                    wins += 1
                else:
                    losses += 1

    total = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": (wins + 0.5 * draws) / max(total, 1),
        "baseline": baseline_type,
        "total": total,
    }
