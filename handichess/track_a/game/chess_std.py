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

    State representation: the numpy state array stores the starting FEN
    and the full move history (as UCI strings), separated by a newline:

        START_FEN\\nMOVE1 MOVE2 MOVE3 ...

    This preserves the complete move stack, which is required for:
      - is_repetition() → repetition planes in encode_board
      - outcome() → threefold repetition draw detection
    """

    # Generous upper bound: FEN (~90 bytes) + 512 moves × ~5 bytes each
    _MAX_STATE_LEN = 4096

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
        self._state_cache: dict[str, chess.Board] = {}

    def get_init_board(self) -> np.ndarray:
        """Return the initial board state."""
        return self._encode_state(self.start_fen, [])

    def get_board_size(self) -> tuple[int, int]:
        return self._board_size

    def get_action_size(self) -> int:
        return self._action_size

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        b = self._state_to_board(board)
        move = decode_action(action, b)
        assert move in b.legal_moves, f"Illegal move: {move} in position {b.fen()}"
        
        start_fen, moves = self._decode_state(board)
        moves.append(move.uci())
        new_state = self._encode_state(start_fen, moves)
        
        # Incrementally update cache for the new state
        new_key = bytes(new_state[new_state > 0].tolist()).decode("ascii")
        b_next = b.copy()
        b_next.push(move)
        
        if len(self._state_cache) > 100000:
            self._state_cache.clear()
        self._state_cache[new_key] = b_next
        
        next_player = -player  # Alternating turns
        return new_state, next_player

    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        b = self._state_to_board(board)
        return get_legal_move_mask(b)

    def get_game_ended(self, board: np.ndarray, player: int) -> float:
        b = self._state_to_board(board)

        # Check half-move count for max_moves limit
        if b.fullmove_number * 2 - (1 if b.turn == chess.WHITE else 0) >= self.max_moves:
            return 1e-4  # Draw by max moves

        outcome = b.outcome(claim_draw=True)
        if outcome is None:
            return 0.0  # Game not over

        if outcome.winner is None:
            return 1e-4  # Draw

        # Convention: player=+1 is white, player=-1 is black.
        winner_is_white = outcome.winner == chess.WHITE
        if (winner_is_white and player == 1) or (not winner_is_white and player == -1):
            return 1.0
        else:
            return -1.0

    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        No-op canonical form for chess.

        Rationale: encode_board() already handles perspective correctly:
          - Planes 0-5 = current player's pieces (via is_own = piece.color == turn)
          - Planes 6-11 = opponent's pieces
          - Plane 14 = color (white=1, black=0)

        Mirroring the board would require remapping all action indices
        (encode/decode, valid_moves mask) to the mirrored coordinate
        system. Without that remapping, the net's policy (in mirrored
        space) gets applied to actual-coordinate actions, silently
        corrupting all black-side nodes.

        The cost of no-op canonical: the net must learn both board
        orientations. This is slightly less sample-efficient but
        eliminates a class of subtle bugs entirely.
        """
        return board.copy()

    def get_encoded_state(self, board: np.ndarray) -> np.ndarray:
        """Encode the board as input planes for the neural network."""
        b = self._state_to_board(board)
        return encode_board(b)

    def string_representation(self, board: np.ndarray) -> str:
        """FEN of the current position (unique per position, not per path)."""
        b = self._state_to_board(board)
        return b.fen()

    def display(self, board: np.ndarray) -> None:
        b = self._state_to_board(board)
        print(b)
        print(f"FEN: {b.fen()}")
        print()

    # ── Internal helpers ────────────────────────────────────────────

    def _encode_state(self, start_fen: str, moves: list[str]) -> np.ndarray:
        """Pack start FEN + move list into a fixed-size numpy array."""
        arr = np.zeros(self._MAX_STATE_LEN, dtype=np.uint8)
        text = start_fen
        if moves:
            text += "\n" + " ".join(moves)
        text_bytes = text.encode("ascii")
        arr[: len(text_bytes)] = list(text_bytes)
        return arr

    def _decode_state(self, arr: np.ndarray) -> tuple[str, list[str]]:
        """Unpack numpy array → (start_fen, move_list)."""
        text = bytes(arr[arr > 0].tolist()).decode("ascii")
        parts = text.split("\n", 1)
        start_fen = parts[0]
        if len(parts) > 1 and parts[1].strip():
            moves = parts[1].strip().split()
        else:
            moves = []
        return start_fen, moves

    def _replay_board(self, start_fen: str, moves: list[str]) -> chess.Board:
        """Reconstruct a Board by replaying moves from start FEN."""
        board = chess.Board(start_fen)
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        return board

    def _state_to_board(self, arr: np.ndarray) -> chess.Board:
        """Convert numpy state array to a python-chess Board with full history."""
        key = bytes(arr[arr > 0].tolist()).decode("ascii")
        if key in self._state_cache:
            return self._state_cache[key]
            
        start_fen, moves = self._decode_state(arr)
        board = self._replay_board(start_fen, moves)
        
        if len(self._state_cache) > 100000:
            self._state_cache.clear()
        self._state_cache[key] = board
        return board

    # ── Factory methods ─────────────────────────────────────────────

    @classmethod
    def from_matchup(
        cls,
        pattern_id: str,
        noq_color: str = "white",
        max_moves: int = 512,
    ) -> "ChessGame":
        """
        Create a ChessGame with a match-up starting position.

        Args:
            pattern_id: Match-up pattern ID (e.g. "rook_bishop_pawn").
            noq_color: "white" or "black" - which side plays without the Queen.
            max_moves: Maximum half-moves.

        Returns:
            ChessGame instance with the match-up FEN.
        """
        from handichess.common.handicap import (
            get_pattern_by_id,
            generate_position,
        )
        import chess as chess_module

        side = chess_module.WHITE if noq_color == "white" else chess_module.BLACK
        pattern = get_pattern_by_id(pattern_id)
        pos = generate_position(pattern, side)
        return cls(start_fen=pos.fen, max_moves=max_moves)
