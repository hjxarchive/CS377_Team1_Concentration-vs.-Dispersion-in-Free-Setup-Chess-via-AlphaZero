"""
Self-play pipeline for AlphaZero training.

Generates training data by having the neural network play against itself.
Games are initialized from handicap positions (sampled from configured patterns).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .game.base import Game
from .net import AlphaZeroNet
from .mcts import MCTS


@dataclass
class TrainingExample:
    """A single training example from self-play."""
    encoded_state: np.ndarray   # (planes, h, w) — canonical form
    policy_target: np.ndarray   # (action_size,) — MCTS visit probabilities
    value_target: float         # Game outcome from this state's perspective: +1, 0, -1


class ReplayBuffer:
    """Fixed-size replay buffer for training examples."""

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, examples: list[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class SelfPlay:
    """
    Self-play game generator.

    Plays games using MCTS + neural network, collecting (state, policy, value)
    training examples. Games start from handicap positions.
    """

    def __init__(
        self,
        game: Game,
        net: AlphaZeroNet,
        mcts_config: Optional[dict] = None,
        selfplay_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.game = game
        self.net = net
        self.device = device

        # MCTS config
        mc = mcts_config or {}
        self.num_simulations = mc.get("num_simulations", 800)
        self.c_puct = mc.get("c_puct", 1.25)
        self.dirichlet_alpha = mc.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = mc.get("dirichlet_epsilon", 0.25)

        # Self-play config
        sc = selfplay_config or {}
        self.max_moves = sc.get("max_moves", 512)
        self.temperature_threshold = mc.get("temperature_threshold", 30)
        self.exploration_plies = sc.get("exploration_plies", 16)

    def play_game(
        self,
        start_state: Optional[np.ndarray] = None,
    ) -> list[TrainingExample]:
        """
        Play one complete self-play game.

        Args:
            start_state: Optional starting state. If None, uses game default.

        Returns:
            List of training examples from the game.
        """
        mcts = MCTS(
            game=self.game,
            net=self.net,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            device=self.device,
        )

        state = start_state if start_state is not None else self.game.get_init_board()
        player = 1
        move_count = 0

        # Collect (state, policy, player) during game
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        while move_count < self.max_moves:
            # Check if game is over
            result = self.game.get_game_ended(state, player)
            if result != 0:
                break

            # Temperature schedule
            if move_count < self.exploration_plies:
                temperature = 1.0
            else:
                temperature = 0.1  # Near-deterministic

            # MCTS search
            action_probs = mcts.search(
                state, player,
                temperature=temperature,
                add_noise=(move_count < self.exploration_plies),
            )

            # Store canonical state + policy
            canonical = self.game.get_canonical_form(state, player)
            encoded = self.game.get_encoded_state(canonical)

            # Apply symmetries for data augmentation
            symmetries = self.game.get_symmetries(canonical, action_probs)
            for sym_board, sym_pi in symmetries:
                sym_encoded = self.game.get_encoded_state(sym_board)
                trajectory.append((sym_encoded, sym_pi, player))

            # Select action
            action = np.random.choice(len(action_probs), p=action_probs)

            # Apply action
            state, player = self.game.get_next_state(state, player, action)
            move_count += 1

        # Determine game result
        result = self.game.get_game_ended(state, player)
        if result == 0:
            # Max moves reached → draw
            result = 1e-4

        # Create training examples with final outcome
        examples = []
        for encoded, policy, p in trajectory:
            # Value target from this player's perspective
            if abs(result) < 0.01:  # Draw
                value = 0.0
            elif p == player:
                # Same player as the one who ended the game
                value = result
            else:
                value = -result

            examples.append(TrainingExample(
                encoded_state=encoded,
                policy_target=policy,
                value_target=value,
            ))

        return examples

    def generate_games(
        self,
        num_games: int,
        start_states: Optional[list[np.ndarray]] = None,
    ) -> list[TrainingExample]:
        """
        Generate multiple self-play games and collect all training examples.

        Args:
            num_games: Number of games to play.
            start_states: Optional list of starting states to sample from.
                          If provided, each game randomly picks one.

        Returns:
            Combined list of training examples from all games.
        """
        all_examples = []

        for i in range(num_games):
            if start_states:
                start = random.choice(start_states)
            else:
                start = None

            examples = self.play_game(start_state=start)
            all_examples.extend(examples)

        return all_examples


def create_handicap_start_states(
    game_class,
    pattern_ids: list[str],
    max_moves: int = 512,
) -> list[np.ndarray]:
    """
    Create a list of starting states from handicap positions.
    Both white-handicap and black-handicap positions are included.

    Args:
        game_class: The ChessGame class.
        pattern_ids: List of pattern IDs to use.
        max_moves: Max moves per game.

    Returns:
        List of numpy starting states.
    """
    states = []
    for pid in pattern_ids:
        for side in ["white", "black"]:
            g = game_class.from_handicap(pid, side, max_moves)
            states.append(g.get_init_board())
    return states
