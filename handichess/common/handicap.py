"""
Handicap position generator.

Generates FEN strings for material-handicap chess positions by removing
pieces from the standard starting position according to predefined patterns.
All patterns remove exactly 9 points of material (= 1 queen).
"""

from __future__ import annotations

import chess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── Standard piece point values ──────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Map single-letter piece codes (from YAML) → python-chess piece types
_PIECE_CODE_MAP = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Removal:
    """A single piece removal: piece type + square name (for white side)."""
    piece: int            # chess.PAWN … chess.QUEEN
    square_name: str      # e.g. "d1", "a2"

    @property
    def white_square(self) -> int:
        """Square index when handicap side is white."""
        return chess.parse_square(self.square_name)

    @property
    def black_square(self) -> int:
        """Mirror square for when handicap side is black (flip rank)."""
        sq = self.white_square
        return chess.square_mirror(sq)


@dataclass(frozen=True)
class RemovalPattern:
    """A complete removal pattern (always sums to 9 points)."""
    pattern_id: str
    description: str
    total_points: int
    num_removed: int
    concentration: str     # "highest" | "high" | "medium" | "low"
    phase: int             # 1 or 2
    removals: tuple[Removal, ...]

    def material_diff_vector(self) -> dict[str, int]:
        """
        Returns the count of each piece type removed.
        Keys: "Q", "R", "B", "N", "P".  Values are non-negative integers.
        """
        counts = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
        code_map_inv = {v: k for k, v in _PIECE_CODE_MAP.items() if k != "K"}
        for r in self.removals:
            code = code_map_inv[r.piece]
            counts[code] += 1
        return counts


@dataclass
class HandicapPosition:
    """A concrete handicap starting position."""
    pattern_id: str
    handicap_side: chess.Color   # chess.WHITE or chess.BLACK
    fen: str
    material_diff: dict[str, int]   # {"Q": 0, "R": 1, "B": 1, ...}

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "handicap_side": "white" if self.handicap_side == chess.WHITE else "black",
            "fen": self.fen,
            "material_diff": self.material_diff,
        }


# ── Pattern loading ──────────────────────────────────────────────────────────
def _load_patterns_from_yaml(yaml_path: Optional[Path] = None) -> list[RemovalPattern]:
    """Load removal patterns from the YAML config file."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "config" / "patterns.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    patterns = []
    for pid, pdata in data["patterns"].items():
        removals = []
        for r in pdata["removals"]:
            piece_type = _PIECE_CODE_MAP[r["piece"]]
            removals.append(Removal(piece=piece_type, square_name=r["square"]))

        pattern = RemovalPattern(
            pattern_id=pid,
            description=pdata["description"],
            total_points=pdata["total_points"],
            num_removed=pdata["num_removed"],
            concentration=pdata["concentration"],
            phase=pdata["phase"],
            removals=tuple(removals),
        )
        # Sanity check: total points must equal sum of piece values
        actual_sum = sum(PIECE_VALUES[r.piece] for r in removals)
        assert actual_sum == pattern.total_points, (
            f"Pattern '{pid}': declared {pattern.total_points} pts "
            f"but removals sum to {actual_sum}"
        )
        patterns.append(pattern)

    return patterns


# ── Module-level cache ───────────────────────────────────────────────────────
_PATTERNS: Optional[list[RemovalPattern]] = None


def get_patterns(phase: Optional[int] = None) -> list[RemovalPattern]:
    """
    Get all removal patterns, optionally filtered by phase.

    Args:
        phase: If given, return only patterns for that phase (1 or 2).

    Returns:
        List of RemovalPattern objects.
    """
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = _load_patterns_from_yaml()

    if phase is not None:
        return [p for p in _PATTERNS if p.phase == phase]
    return list(_PATTERNS)


def get_pattern_by_id(pattern_id: str) -> RemovalPattern:
    """Look up a single pattern by its ID."""
    for p in get_patterns():
        if p.pattern_id == pattern_id:
            return p
    raise ValueError(f"Unknown pattern: {pattern_id}")


# ── Position generation ──────────────────────────────────────────────────────
def make_handicap_board(
    pattern: RemovalPattern,
    handicap_side: chess.Color,
) -> chess.Board:
    """
    Create a chess.Board with pieces removed according to the pattern.

    The board starts from the standard position, then removes the specified
    pieces from the handicap side.  All standard rules (castling rights, etc.)
    are updated automatically by python-chess.

    Args:
        pattern: Which pieces to remove.
        handicap_side: chess.WHITE or chess.BLACK — who is handicapped.

    Returns:
        A python-chess Board ready to play from.
    """
    board = chess.Board()  # standard starting position

    for removal in pattern.removals:
        if handicap_side == chess.WHITE:
            sq = removal.white_square
        else:
            sq = removal.black_square

        # Verify the expected piece is actually on that square
        piece = board.piece_at(sq)
        expected_piece = chess.Piece(removal.piece, handicap_side)
        assert piece == expected_piece, (
            f"Pattern '{pattern.pattern_id}': expected {expected_piece} at "
            f"{chess.square_name(sq)}, found {piece}"
        )

        board.remove_piece_at(sq)

    # Update castling rights — python-chess handles this via set_fen,
    # but after remove_piece_at we should clean up manually.
    board.set_castling_fen(_compute_castling_fen(board))

    return board


def _compute_castling_fen(board: chess.Board) -> str:
    """Recompute castling FEN based on actual rook/king positions."""
    castling = ""

    # White
    wk = board.king(chess.WHITE)
    if wk == chess.E1:
        if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            castling += "K"
        if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            castling += "Q"

    # Black
    bk = board.king(chess.BLACK)
    if bk == chess.E8:
        if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            castling += "k"
        if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            castling += "q"

    return castling if castling else "-"


def generate_position(
    pattern: RemovalPattern,
    handicap_side: chess.Color,
) -> HandicapPosition:
    """
    Generate a single HandicapPosition for a given pattern and side.

    Args:
        pattern: The removal pattern.
        handicap_side: Which side is handicapped.

    Returns:
        HandicapPosition with FEN and metadata.
    """
    board = make_handicap_board(pattern, handicap_side)
    return HandicapPosition(
        pattern_id=pattern.pattern_id,
        handicap_side=handicap_side,
        fen=board.fen(),
        material_diff=pattern.material_diff_vector(),
    )


def generate_all_positions(
    phase: Optional[int] = None,
) -> list[HandicapPosition]:
    """
    Generate all handicap positions (both colors) for all patterns.

    Args:
        phase: If given, only generate positions for that phase.

    Returns:
        List of HandicapPosition objects (2 per pattern: white and black).
    """
    positions = []
    for pattern in get_patterns(phase=phase):
        for side in [chess.WHITE, chess.BLACK]:
            positions.append(generate_position(pattern, side))
    return positions


# ── Utilities ────────────────────────────────────────────────────────────────
def count_material(board: chess.Board, color: chess.Color) -> int:
    """Count total material points for a side (excluding king)."""
    total = 0
    for piece_type in PIECE_VALUES:
        total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total


def material_balance(board: chess.Board) -> int:
    """Material advantage for white (white - black), in standard points."""
    return count_material(board, chess.WHITE) - count_material(board, chess.BLACK)


# ── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Handicap Positions ===\n")
    for pos in generate_all_positions():
        side_str = "WHITE" if pos.handicap_side == chess.WHITE else "BLACK"
        print(f"[{pos.pattern_id}] handicap={side_str}")
        print(f"  FEN: {pos.fen}")
        print(f"  Removed: {pos.material_diff}")
        print()
