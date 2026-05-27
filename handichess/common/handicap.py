"""
Matchup position generator (Design B).

Generates FEN strings for NoQ vs Q match-ups.
Each pattern specifies 9 points of material removed for both sides:
- NoQ side: removes Queen (9 pts).
- Q side: removes a Bundle (9 pts).
Both sides end up with exactly 30 points of material.
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
        """Square index when the piece is removed from white."""
        return chess.parse_square(self.square_name)

    @property
    def black_square(self) -> int:
        """Mirror square for when the piece is removed from black (flip rank)."""
        sq = self.white_square
        return chess.square_mirror(sq)


@dataclass(frozen=True)
class MatchupPattern:
    """A complete match-up pattern (both sides remove 9 points)."""
    pattern_id: str
    description: str
    total_points: int      # Usually 9
    num_removed: int       # Number of pieces in the bundle
    concentration: str     # "highest" | "high" | "medium" | "low"
    phase: int             # 1 or 2
    noq_removals: tuple[Removal, ...]  # Always Queen
    q_removals: tuple[Removal, ...]    # The Bundle

    def bundle_vector(self) -> dict[str, int]:
        """
        Returns the count of each piece type in the Bundle (q_removals).
        Keys: "Q", "R", "B", "N", "P".
        """
        counts = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
        code_map_inv = {v: k for k, v in _PIECE_CODE_MAP.items() if k != "K"}
        for r in self.q_removals:
            code = code_map_inv[r.piece]
            counts[code] += 1
        return counts


@dataclass
class MatchupPosition:
    """A concrete match-up starting position."""
    pattern_id: str
    noq_color: chess.Color       # chess.WHITE or chess.BLACK
    fen: str
    bundle_vector: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "noq_color": "white" if self.noq_color == chess.WHITE else "black",
            "fen": self.fen,
            "bundle_vector": self.bundle_vector,
        }


# ── Pattern loading ──────────────────────────────────────────────────────────
def _load_patterns_from_yaml(yaml_path: Optional[Path] = None) -> list[MatchupPattern]:
    """Load match-up patterns from the YAML config file."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "config" / "patterns.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    patterns = []
    for pid, pdata in data["patterns"].items():
        noq_rems = []
        for r in pdata.get("noq_removals", []):
            piece_type = _PIECE_CODE_MAP[r["piece"]]
            noq_rems.append(Removal(piece=piece_type, square_name=r["square"]))

        q_rems = []
        for r in pdata.get("q_removals", []):
            piece_type = _PIECE_CODE_MAP[r["piece"]]
            q_rems.append(Removal(piece=piece_type, square_name=r["square"]))

        pattern = MatchupPattern(
            pattern_id=pid,
            description=pdata["description"],
            total_points=pdata["total_points"],
            num_removed=pdata["num_removed"],
            concentration=pdata["concentration"],
            phase=pdata["phase"],
            noq_removals=tuple(noq_rems),
            q_removals=tuple(q_rems),
        )

        # Sanity check: both removals must sum to 9 points
        noq_sum = sum(PIECE_VALUES[r.piece] for r in noq_rems)
        q_sum = sum(PIECE_VALUES[r.piece] for r in q_rems)
        
        assert noq_sum == 9, f"Pattern '{pid}': NoQ removals sum to {noq_sum}, expected 9"
        assert q_sum == 9, f"Pattern '{pid}': Q removals sum to {q_sum}, expected 9"

        patterns.append(pattern)

    return patterns


# ── Module-level cache ───────────────────────────────────────────────────────
_PATTERNS: Optional[list[MatchupPattern]] = None


def get_patterns(phase: Optional[int] = None) -> list[MatchupPattern]:
    """
    Get all match-up patterns, optionally filtered by phase.
    """
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = _load_patterns_from_yaml()

    if phase is not None:
        return [p for p in _PATTERNS if p.phase == phase]
    return list(_PATTERNS)


def get_pattern_by_id(pattern_id: str) -> MatchupPattern:
    """Look up a single pattern by its ID."""
    for p in get_patterns():
        if p.pattern_id == pattern_id:
            return p
    raise ValueError(f"Unknown pattern: {pattern_id}")


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


# ── Position generation ──────────────────────────────────────────────────────
def make_matchup_board(
    pattern: MatchupPattern,
    noq_color: chess.Color,
) -> chess.Board:
    """
    Create a chess.Board for a NoQ vs Q match-up.

    Args:
        pattern: The MatchupPattern specifying noq_removals and q_removals.
        noq_color: Which color plays the NoQ side (removes noq_removals).
                   The opposite color plays the Q side (removes q_removals).

    Returns:
        A python-chess Board ready to play from.
    """
    board = chess.Board()  # standard starting position

    # Determine colors
    q_color = not noq_color

    # 1. Apply NoQ removals
    for removal in pattern.noq_removals:
        sq = removal.white_square if noq_color == chess.WHITE else removal.black_square
        piece = board.piece_at(sq)
        expected_piece = chess.Piece(removal.piece, noq_color)
        assert piece == expected_piece, (
            f"Pattern '{pattern.pattern_id}': expected {expected_piece} at "
            f"{chess.square_name(sq)}, found {piece}"
        )
        board.remove_piece_at(sq)

    # 2. Apply Q (Bundle) removals
    for removal in pattern.q_removals:
        sq = removal.white_square if q_color == chess.WHITE else removal.black_square
        piece = board.piece_at(sq)
        expected_piece = chess.Piece(removal.piece, q_color)
        assert piece == expected_piece, (
            f"Pattern '{pattern.pattern_id}': expected {expected_piece} at "
            f"{chess.square_name(sq)}, found {piece}"
        )
        board.remove_piece_at(sq)

    # 3. Validation: Both sides must have exactly 30 material points.
    assert count_material(board, chess.WHITE) == 30, "White material is not 30."
    assert count_material(board, chess.BLACK) == 30, "Black material is not 30."

    # Update castling rights
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
    pattern: MatchupPattern,
    noq_color: chess.Color,
) -> MatchupPosition:
    """
    Generate a single MatchupPosition for a given pattern and noq_color.
    """
    board = make_matchup_board(pattern, noq_color)
    return MatchupPosition(
        pattern_id=pattern.pattern_id,
        noq_color=noq_color,
        fen=board.fen(),
        bundle_vector=pattern.bundle_vector(),
    )


def generate_all_positions(
    phase: Optional[int] = None,
) -> list[MatchupPosition]:
    """
    Generate all match-up positions (both colors) for all patterns.
    """
    positions = []
    for pattern in get_patterns(phase=phase):
        for side in [chess.WHITE, chess.BLACK]:
            positions.append(generate_position(pattern, side))
    return positions


# ── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Matchup Positions ===")
    for pos in generate_all_positions():
        color_str = "WHITE" if pos.noq_color == chess.WHITE else "BLACK"
        print(f"[{pos.pattern_id}] NoQ_Color={color_str}")
        print(f"  FEN: {pos.fen}")
        print(f"  Bundle removed from Q-side: {pos.bundle_vector}")
        print()
