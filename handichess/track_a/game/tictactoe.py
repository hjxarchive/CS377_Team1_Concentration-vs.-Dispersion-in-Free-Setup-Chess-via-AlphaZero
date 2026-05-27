"""
TicTacToe implementation of the Game ABC.

Used as a verification target: the AlphaZero core should learn perfect
play (never lose) on this simple game before moving to chess.
"""

from __future__ import annotations

import numpy as np

from .base import Game


class TicTacToeGame(Game):
    """
    3×3 TicTacToe.

    Board representation: 3×3 numpy array.
      0 = empty, +1 = player 1 (X), -1 = player 2 (O).

    Action space: 9 positions (row * 3 + col).
    """

    def __init__(self):
        self._board_size = (3, 3)
        self._action_size = 9

    def get_init_board(self) -> np.ndarray:
        return np.zeros(self._board_size, dtype=np.float32)

    def get_board_size(self) -> tuple[int, int]:
        return self._board_size

    def get_action_size(self) -> int:
        return self._action_size

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        b = board.copy()
        row, col = divmod(action, 3)
        assert b[row, col] == 0, f"Illegal move: position ({row},{col}) is occupied"
        b[row, col] = player
        return b, -player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        valid = np.zeros(self._action_size, dtype=np.float32)
        for i in range(9):
            row, col = divmod(i, 3)
            if board[row, col] == 0:
                valid[i] = 1.0
        return valid

    def get_game_ended(self, board: np.ndarray, player: int) -> float:
        # Check rows, cols, diagonals
        for i in range(3):
            # Rows
            if abs(board[i].sum()) == 3:
                return 1.0 if board[i, 0] == player else -1.0
            # Columns
            if abs(board[:, i].sum()) == 3:
                return 1.0 if board[0, i] == player else -1.0

        # Diagonals
        diag1 = board[0, 0] + board[1, 1] + board[2, 2]
        if abs(diag1) == 3:
            return 1.0 if board[1, 1] == player else -1.0

        diag2 = board[0, 2] + board[1, 1] + board[2, 0]
        if abs(diag2) == 3:
            return 1.0 if board[1, 1] == player else -1.0

        # Draw (board full, no winner)
        if not np.any(board == 0):
            return 1e-4

        # Game not over
        return 0.0

    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        return board * player

    def get_encoded_state(self, board: np.ndarray) -> np.ndarray:
        """
        Encode as 3 planes:
          plane 0: current player's pieces (1 where board > 0)
          plane 1: opponent's pieces (1 where board < 0)
          plane 2: empty squares (1 where board == 0)

        Note: board should already be in canonical form.
        """
        encoded = np.zeros((3, 3, 3), dtype=np.float32)
        encoded[0] = (board > 0).astype(np.float32)
        encoded[1] = (board < 0).astype(np.float32)
        encoded[2] = (board == 0).astype(np.float32)
        return encoded

    def string_representation(self, board: np.ndarray) -> str:
        return board.tobytes().hex()

    def get_symmetries(
        self, board: np.ndarray, pi: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        TicTacToe has 8 symmetries (4 rotations × 2 reflections).
        """
        assert len(pi) == self._action_size
        pi_board = pi.reshape(3, 3)
        symmetries = []

        for rot in range(4):
            b_rot = np.rot90(board, rot)
            p_rot = np.rot90(pi_board, rot)
            symmetries.append((b_rot.copy(), p_rot.flatten().copy()))
            # Flip
            b_flip = np.fliplr(b_rot)
            p_flip = np.fliplr(p_rot)
            symmetries.append((b_flip.copy(), p_flip.flatten().copy()))

        return symmetries

    def display(self, board: np.ndarray) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        print("  0 1 2")
        for i in range(3):
            row_str = " ".join(symbols[int(board[i, j])] for j in range(3))
            print(f"{i} {row_str}")
        print()
