"""
Board and action encoding for chess, following the AlphaZero scheme.

Board encoding:
  - 12 piece planes (P, N, B, R, Q, K for each color)
  - 2 repetition planes (1-fold, 2-fold repetition)
  - 1 color plane (all 1s if white to move, all 0s if black)
  - 1 total move count (normalized)
  - 4 castling rights (K, Q, k, q)
  - 1 no-progress count (normalized fifty-move counter)
  Total: 21 input planes of 8×8

Action encoding (AlphaZero 8×8×73):
  - 56 "queen moves" (7 distances × 8 directions)
  - 8 knight moves
  - 9 underpromotions (3 piece types × 3 directions)
  Total: 73 planes, from-square determines the 8×8 position.

All encoding is from the perspective of the current player (canonical form).
"""

from __future__ import annotations

import chess
import numpy as np
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────
NUM_INPUT_PLANES = 21
ACTION_PLANES = 73
ACTION_SIZE = 8 * 8 * ACTION_PLANES  # 4672

# Piece type to plane index (current player's pieces first)
_PIECE_PLANES = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 1,
    (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,
    (chess.QUEEN, True): 4,
    (chess.KING, True): 5,
    (chess.PAWN, False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,
    (chess.QUEEN, False): 10,
    (chess.KING, False): 11,
}

# Queen-move direction vectors (file_delta, rank_delta)
# 8 directions: N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

# Knight move deltas
_KNIGHT_MOVES = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion pieces (queen promotion is encoded as queen-move)
_UNDERPROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Direction indices for underpromotions: left-capture, straight, right-capture
_UNDERPROMOTION_DIRS = [(-1, 1), (0, 1), (1, 1)]


# ── Board encoding ──────────────────────────────────────────────────────────
def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as a (21, 8, 8) float32 tensor.

    The board should be in canonical form (current player = white's perspective).

    Args:
        board: A python-chess Board.

    Returns:
        numpy array of shape (NUM_INPUT_PLANES, 8, 8).
    """
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    turn = board.turn

    # Piece planes (0-11)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            is_own = piece.color == turn
            plane_idx = _PIECE_PLANES[(piece.piece_type, is_own)]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            planes[plane_idx, rank, file] = 1.0

    # Repetition planes (12-13)
    # Check if current position has occurred before
    if board.is_repetition(1):
        planes[12, :, :] = 1.0
    if board.is_repetition(2):
        planes[13, :, :] = 1.0

    # Color plane (14): 1 if white to move
    if turn == chess.WHITE:
        planes[14, :, :] = 1.0

    # Move count (15): normalized by 200
    planes[15, :, :] = min(board.fullmove_number / 200.0, 1.0)

    # Castling rights (16-19)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[16, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[17, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[18, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[19, :, :] = 1.0

    # No-progress count (20): fifty-move counter normalized by 100
    planes[20, :, :] = board.halfmove_clock / 100.0

    return planes


# ── Action encoding ──────────────────────────────────────────────────────────
def encode_action(move: chess.Move, board: chess.Board) -> int:
    """
    Encode a chess move into an action index in [0, 4672).

    The action index is: from_square_rank * 8 * 73 + from_square_file * 73 + move_plane

    where move_plane encodes the type and direction of the move.

    Args:
        move: A python-chess Move.
        board: The board state (needed for context, e.g. turn).

    Returns:
        Integer action index.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)

    d_file = to_file - from_file
    d_rank = to_rank - from_rank

    plane = _get_move_plane(d_file, d_rank, move.promotion, board.piece_type_at(from_sq))

    return (from_rank * 8 + from_file) * ACTION_PLANES + plane


def decode_action(action: int, board: chess.Board) -> chess.Move:
    """
    Decode an action index back to a chess move.

    Args:
        action: Integer action index in [0, 4672).
        board: The current board state.

    Returns:
        A python-chess Move.
    """
    sq_idx = action // ACTION_PLANES
    plane = action % ACTION_PLANES
    from_rank = sq_idx // 8
    from_file = sq_idx % 8
    from_sq = chess.square(from_file, from_rank)

    d_file, d_rank, promotion = _decode_move_plane(plane)

    to_file = from_file + d_file
    to_rank = from_rank + d_rank
    to_sq = chess.square(to_file, to_rank)

    # Handle queen promotion for pawn reaching last rank
    piece = board.piece_type_at(from_sq)
    if piece == chess.PAWN and promotion is None:
        if (board.turn == chess.WHITE and to_rank == 7) or \
           (board.turn == chess.BLACK and to_rank == 0):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def _get_move_plane(
    d_file: int, d_rank: int,
    promotion: Optional[int],
    piece_type: Optional[int],
) -> int:
    """
    Determine the move plane index (0-72) from move deltas and context.

    Planes 0-55:  Queen moves (7 distances × 8 directions)
    Planes 56-63: Knight moves (8 moves)
    Planes 64-72: Underpromotions (3 pieces × 3 directions)
    """
    # Underpromotion
    if promotion is not None and promotion != chess.QUEEN:
        piece_idx = _UNDERPROMOTION_PIECES.index(promotion)
        for dir_idx, (df, dr) in enumerate(_UNDERPROMOTION_DIRS):
            if d_file == df and d_rank == dr:
                return 64 + piece_idx * 3 + dir_idx
        # Black pawn promotes going down
        for dir_idx, (df, dr) in enumerate(_UNDERPROMOTION_DIRS):
            if d_file == df and d_rank == -dr:
                return 64 + piece_idx * 3 + dir_idx
        raise ValueError(f"Cannot encode underpromotion: df={d_file}, dr={d_rank}, prom={promotion}")

    # Knight move
    if piece_type == chess.KNIGHT or (d_file, d_rank) in _KNIGHT_MOVES:
        try:
            knight_idx = _KNIGHT_MOVES.index((d_file, d_rank))
            return 56 + knight_idx
        except ValueError:
            pass

    # Queen move (includes pawn, bishop, rook, queen, king normal moves)
    distance = max(abs(d_file), abs(d_rank))
    assert distance >= 1, f"Zero-distance move: df={d_file}, dr={d_rank}"

    # Normalize to unit direction
    unit_f = (d_file // distance) if d_file != 0 else 0
    unit_r = (d_rank // distance) if d_rank != 0 else 0

    try:
        dir_idx = _QUEEN_DIRECTIONS.index((unit_f, unit_r))
    except ValueError:
        raise ValueError(
            f"Cannot find direction for ({unit_f}, {unit_r}) "
            f"from delta ({d_file}, {d_rank})"
        )

    return dir_idx * 7 + (distance - 1)


def _decode_move_plane(plane: int) -> tuple[int, int, Optional[int]]:
    """
    Decode a move plane index to (d_file, d_rank, promotion).

    Returns:
        (d_file, d_rank, promotion) where promotion is None or a piece type.
    """
    if plane < 56:
        # Queen move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        d_file, d_rank = _QUEEN_DIRECTIONS[dir_idx]
        return d_file * distance, d_rank * distance, None

    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        d_file, d_rank = _KNIGHT_MOVES[knight_idx]
        return d_file, d_rank, None

    else:
        # Underpromotion
        idx = plane - 64
        piece_idx = idx // 3
        dir_idx = idx % 3
        d_file, d_rank = _UNDERPROMOTION_DIRS[dir_idx]
        promotion = _UNDERPROMOTION_PIECES[piece_idx]
        return d_file, d_rank, promotion


# ── Legal move mask ──────────────────────────────────────────────────────────
def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Generate a binary mask for all legal moves.

    Args:
        board: Current chess board.

    Returns:
        1D numpy array of shape (ACTION_SIZE,) with 1.0 for legal moves.
    """
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        try:
            action = encode_action(move, board)
            mask[action] = 1.0
        except (ValueError, AssertionError):
            # Skip moves that can't be encoded (shouldn't happen in standard chess)
            pass
    return mask


def encode_action_batch(
    moves: list[chess.Move], board: chess.Board
) -> list[int]:
    """Encode a batch of moves."""
    return [encode_action(m, board) for m in moves]
