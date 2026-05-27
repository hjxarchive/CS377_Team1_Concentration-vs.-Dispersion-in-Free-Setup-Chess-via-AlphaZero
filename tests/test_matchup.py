import pytest
import chess
from handichess.common.handicap import (
    get_patterns,
    make_matchup_board,
    count_material,
    _PIECE_CODE_MAP,
)

def _get_piece_counts(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Return piece counts by letter (e.g. 'Q', 'R') for a specific color."""
    code_map_inv = {v: k for k, v in _PIECE_CODE_MAP.items() if k != "K"}
    counts = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
        code = code_map_inv[piece_type]
        counts[code] = len(board.pieces(piece_type, color))
    return counts

def test_matchup_isolation_invariants():
    """
    Test the fundamental isolation invariants of Design B (NoQ vs Q).
    1. NoQ side has 0 Queens, Q side has 1 Queen.
    2. Both sides have exactly 30 material points.
    3. The per-piece difference vector (NoQ - Q) exactly matches (Bundle - Queen).
    4. The position is valid.
    """
    patterns = get_patterns()
    assert len(patterns) > 0, "Should have loaded some patterns"
    
    for pattern in patterns:
        for noq_color in [chess.WHITE, chess.BLACK]:
            q_color = not noq_color
            board = make_matchup_board(pattern, noq_color)
            
            # Invariant 1: Queen counts
            noq_counts = _get_piece_counts(board, noq_color)
            q_counts = _get_piece_counts(board, q_color)
            
            assert noq_counts["Q"] == 0, f"{pattern.pattern_id} ({noq_color}): NoQ side must have 0 queens"
            assert q_counts["Q"] == 1, f"{pattern.pattern_id} ({noq_color}): Q side must have 1 queen"
            
            # Invariant 2: Total material is 30 for both sides
            assert count_material(board, noq_color) == 30
            assert count_material(board, q_color) == 30
            
            # Invariant 3: Difference Vector
            # NoQ - Q = Bundle - Queen
            bundle_vector = pattern.bundle_vector()
            
            for piece_code in ["Q", "R", "B", "N", "P"]:
                actual_diff = noq_counts[piece_code] - q_counts[piece_code]
                
                # Expected diff:
                # Bundle - Queen
                # If piece is Q, expected diff is -1 (since bundle doesn't contain Q, 0 - 1 = -1)
                # If piece is in Bundle, expected diff is bundle_vector[piece] - 0
                expected_diff = bundle_vector[piece_code]
                if piece_code == "Q":
                    expected_diff -= 1
                    
                assert actual_diff == expected_diff, (
                    f"Vector mismatch for {piece_code} in {pattern.pattern_id} ({noq_color}): "
                    f"actual={actual_diff}, expected={expected_diff}"
                )
            
            # Invariant 4: Board validity
            assert board.is_valid(), f"Board for {pattern.pattern_id} ({noq_color}) is invalid"

if __name__ == "__main__":
    pytest.main([__file__])
