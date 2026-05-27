"""
Standard Chess implementation of the Game ABC.

Wraps python-chess for full standard rules (castling, en passant, promotion,
all draw rules). The only customization is the initial position: games can
start from a handicap FEN instead of the standard position.

Action encoding follows AlphaZero's 8×8×73 scheme.
"""

from __future__ import annotations

from typing import Optional

import chess
import numpy as np

from .base import Game
from ..encoding import (
    encode_board,
    encode_action,
    decode_action,
    get_legal_move_mask,
    NUM_INPUT_PLANES,
    ACTION_PLANES,
)


class ChessGame(Game):
    """
    Standard chess with optional handicap starting position.

    The state is stored as a python-chess Board object serialized to FEN.
    For efficiency, we cache the Board object alongside the numpy state.

    Board numpy representation: we store the FEN string encoded as bytes
    in a fixed-size array. The actual neural network input is produced
    by get_encoded_state().
    """

    # Maximum FEN length (generous upper bound)
    _MAX_FEN_LEN = 256

    def __init__(
        self,
        start_fen: Optional[str] = None,
        max_moves: int = 512,
    ):
        """
        Args:
            start_fen: Starting FEN. None = standard starting position.
            max_moves: Maximum half-moves before forced draw.
        """
        self.start_fen = start_fen or chess.STARTING_FEN
        self.max_moves = max_moves
        self._board_size = (8, 8)
        self._action_size = 8 * 8 * ACTION_PLANES  # 4672

    def get_init_board(self) -> np.ndarray:
        """Return the initial board state as a FEN-encoded array."""
        return self._fen_to_array(self.start_fen)

    def get_board_size(self) -> tuple[int, int]:
        return self._board_size

    def get_action_size(self) -> int:
        return self._action_size

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        b = self._array_to_board(board)
        move = decode_action(action, b)
        assert move in b.legal_moves, f"Illegal move: {move} in position {b.fen()}"
        b.push(move)
        next_player = -player  # Alternating turns
        return self._fen_to_array(b.fen()), next_player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        b = self._array_to_board(board)
        return get_legal_move_mask(b)

    def get_game_ended(self, board: np.ndarray, player: int) -> float:
        b = self._array_to_board(board)

        # Check half-move count for max_moves limit
        if b.fullmove_number * 2 - (1 if b.turn == chess.WHITE else 0) >= self.max_moves:
            return 1e-4  # Draw by max moves

        outcome = b.outcome()
        if outcome is None:
            return 0.0  # Game not over

        if outcome.winner is None:
            return 1e-4  # Draw

        # Determine from current player's perspective
        # player +1 = white at the start, but due to canonical form,
        # the "current player" is whoever's turn it is in the board state.
        # Since we track player separately, we need to check carefully.
        #
        # Convention: player=+1 is the side that moved first (white).
        # If white won and player=+1, return +1. If player=-1, return -1.
        winner_is_white = outcome.winner == chess.WHITE
        if (winner_is_white and player == 1) or (not winner_is_white and player == -1):
            return 1.0
        else:
            return -1.0

    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        For chess, canonical form means the board is always viewed from
        the current player's perspective. If player is -1 (black),
        we flip the board and swap piece colors.
        """
        if player == 1:
            return board.copy()

        b = self._array_to_board(board)
        # Flip: mirror the board and swap colors
        b_mirror = b.mirror()
        return self._fen_to_array(b_mirror.fen())

    def get_encoded_state(self, board: np.ndarray) -> np.ndarray:
        """Encode the board as input planes for the neural network."""
        b = self._array_to_board(board)
        return encode_board(b)

    def string_representation(self, board: np.ndarray) -> str:
        return self._array_to_fen(board)

    def display(self, board: np.ndarray) -> None:
        b = self._array_to_board(board)
        print(b)
        print(f"FEN: {b.fen()}")
        print()

    # ── Internal helpers ────────────────────────────────────────────

    def _fen_to_array(self, fen: str) -> np.ndarray:
        """Convert FEN string to a fixed-size numpy array for storage."""
        arr = np.zeros(self._MAX_FEN_LEN, dtype=np.uint8)
        fen_bytes = fen.encode("ascii")
        arr[: len(fen_bytes)] = list(fen_bytes)
        return arr

    def _array_to_fen(self, arr: np.ndarray) -> str:
        """Convert numpy array back to FEN string."""
        fen_bytes = bytes(arr[arr > 0].tolist())
        return fen_bytes.decode("ascii")

    def _array_to_board(self, arr: np.ndarray) -> chess.Board:
        """Convert numpy array to a python-chess Board."""
        fen = self._array_to_fen(arr)
        return chess.Board(fen)

    # ── Factory methods ─────────────────────────────────────────────

    @classmethod
    def from_handicap(
        cls,
        pattern_id: str,
        handicap_side: str = "white",
        max_moves: int = 512,
    ) -> "ChessGame":
        """
        Create a ChessGame with a handicap starting position.

        Args:
            pattern_id: Removal pattern ID (e.g. "queen", "rook_bishop_pawn").
            handicap_side: "white" or "black".
            max_moves: Maximum half-moves.

        Returns:
            ChessGame instance with the handicap FEN.
        """
        from handichess.common.handicap import (
            get_pattern_by_id,
            generate_position,
        )
        import chess as chess_module

        side = chess_module.WHITE if handicap_side == "white" else chess_module.BLACK
        pattern = get_pattern_by_id(pattern_id)
        pos = generate_position(pattern, side)
        return cls(start_fen=pos.fen, max_moves=max_moves)
