"""
Game Abstract Base Class.

Defines the interface that separates the game-agnostic AlphaZero core
(MCTS, neural network, self-play, training) from game-specific logic.
Implementations: TicTacToe (for verification), standard chess (with handicap).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Game(ABC):
    """
    Abstract base class for a two-player zero-sum game.

    Convention:
      - Player 1 = +1, Player 2 = -1.
      - Board state is a numpy array.
      - Canonical form: the board as seen from the current player's perspective.
    """

    @abstractmethod
    def get_init_board(self) -> np.ndarray:
        """Return the initial board state."""
        ...

    @abstractmethod
    def get_board_size(self) -> tuple[int, ...]:
        """Return the board dimensions, e.g. (3,3) for TicTacToe, (8,8) for chess."""
        ...

    @abstractmethod
    def get_action_size(self) -> int:
        """Return the total number of possible actions (including illegal ones)."""
        ...

    @abstractmethod
    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        """
        Apply an action and return the new board and the next player.

        Args:
            board: Current board state.
            player: Current player (+1 or -1).
            action: Action index.

        Returns:
            (next_board, next_player) tuple.
        """
        ...

    @abstractmethod
    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return a binary mask of valid actions.

        Args:
            board: Current board state.
            player: Current player (+1 or -1).

        Returns:
            1D numpy array of shape (action_size,) with 1 for valid moves, 0 otherwise.
        """
        ...

    @abstractmethod
    def get_game_ended(self, board: np.ndarray, player: int) -> float:
        """
        Check if the game has ended.

        Args:
            board: Current board state.
            player: Current player (+1 or -1).

        Returns:
            0 if game is not over.
            +1 if `player` has won.
            -1 if `player` has lost.
            1e-4 (small positive) for draw.
        """
        ...

    @abstractmethod
    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return the board from the perspective of the given player.
        For player +1, return board as-is.
        For player -1, swap pieces / flip perspective.

        This is critical for MCTS: the neural network always evaluates
        from the "current player" perspective.
        """
        ...

    @abstractmethod
    def get_encoded_state(self, board: np.ndarray) -> np.ndarray:
        """
        Encode the board as input planes for the neural network.

        Args:
            board: Board state (already in canonical form).

        Returns:
            numpy array of shape (num_planes, *board_size).
        """
        ...

    @abstractmethod
    def string_representation(self, board: np.ndarray) -> str:
        """
        Return a unique string representation of the board state,
        suitable for use as a dictionary key (e.g. in MCTS node cache).
        """
        ...

    def get_symmetries(
        self, board: np.ndarray, pi: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return a list of (board, policy) pairs representing symmetries.
        Default: no symmetries (just the original).
        Override for games with board symmetries (e.g. TicTacToe rotations).
        """
        return [(board, pi)]

    def display(self, board: np.ndarray) -> None:
        """Print a human-readable board. Optional."""
        print(self.string_representation(board))
