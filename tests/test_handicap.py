"""Tests for the handicap position generator."""

import chess
import pytest

from handichess.common.handicap import (
    get_patterns,
    get_pattern_by_id,
    generate_position,
    generate_all_positions,
    make_handicap_board,
    count_material,
    material_balance,
    PIECE_VALUES,
)


class TestPatternLoading:
    """Test that patterns load correctly from YAML."""

    def test_load_all_patterns(self):
        patterns = get_patterns()
        assert len(patterns) == 8, f"Expected 8 patterns, got {len(patterns)}"

    def test_phase1_patterns(self):
        phase1 = get_patterns(phase=1)
        assert len(phase1) == 4
        ids = {p.pattern_id for p in phase1}
        assert "queen" in ids
        assert "rook_bishop_pawn" in ids
        assert "rook_knight_pawn" in ids
        assert "bishop_bishop_knight" in ids

    def test_phase2_patterns(self):
        phase2 = get_patterns(phase=2)
        assert len(phase2) == 4

    def test_all_patterns_sum_to_9(self):
        for pattern in get_patterns():
            total = sum(
                PIECE_VALUES[r.piece] for r in pattern.removals
            )
            assert total == 9, (
                f"Pattern '{pattern.pattern_id}' sums to {total}, not 9"
            )

    def test_get_pattern_by_id(self):
        p = get_pattern_by_id("queen")
        assert p.pattern_id == "queen"
        assert p.num_removed == 1

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError):
            get_pattern_by_id("nonexistent")


class TestPositionGeneration:
    """Test that handicap positions are generated correctly."""

    def test_queen_removal_white(self):
        pattern = get_pattern_by_id("queen")
        board = make_handicap_board(pattern, chess.WHITE)

        # White should have no queen
        assert len(board.pieces(chess.QUEEN, chess.WHITE)) == 0
        # Black should still have queen
        assert len(board.pieces(chess.QUEEN, chess.BLACK)) == 1
        # White material = 39 - 9 = 30
        assert count_material(board, chess.WHITE) == 30
        # Black material = 39
        assert count_material(board, chess.BLACK) == 39

    def test_queen_removal_black(self):
        pattern = get_pattern_by_id("queen")
        board = make_handicap_board(pattern, chess.BLACK)

        assert len(board.pieces(chess.QUEEN, chess.BLACK)) == 0
        assert len(board.pieces(chess.QUEEN, chess.WHITE)) == 1
        assert count_material(board, chess.BLACK) == 30
        assert count_material(board, chess.WHITE) == 39

    def test_all_patterns_remove_9_points(self):
        """Verify material balance is exactly -9 for all patterns."""
        for pattern in get_patterns():
            for side in [chess.WHITE, chess.BLACK]:
                board = make_handicap_board(pattern, side)
                handicap_material = count_material(board, side)
                full_material = count_material(chess.Board(), side)
                removed = full_material - handicap_material
                assert removed == 9, (
                    f"Pattern '{pattern.pattern_id}' side={side}: "
                    f"removed {removed} points, expected 9"
                )

    def test_material_diff_vector(self):
        pattern = get_pattern_by_id("rook_bishop_pawn")
        diff = pattern.material_diff_vector()
        assert diff["R"] == 1
        assert diff["B"] == 1
        assert diff["P"] == 1
        assert diff["Q"] == 0
        assert diff["N"] == 0

    def test_generate_position_returns_valid_fen(self):
        pattern = get_pattern_by_id("queen")
        pos = generate_position(pattern, chess.WHITE)

        # FEN should be parseable
        board = chess.Board(pos.fen)
        assert board.is_valid()

    def test_generate_all_positions(self):
        positions = generate_all_positions()
        # 8 patterns × 2 sides = 16 positions
        assert len(positions) == 16

    def test_generate_phase1_positions(self):
        positions = generate_all_positions(phase=1)
        # 4 patterns × 2 sides = 8 positions
        assert len(positions) == 8

    def test_castling_rights_updated(self):
        """When a rook is removed, castling rights should be updated."""
        pattern = get_pattern_by_id("rook_bishop_pawn")
        board = make_handicap_board(pattern, chess.WHITE)

        # White a1 rook removed → no queenside castling for white
        assert not board.has_queenside_castling_rights(chess.WHITE)
        # White h1 rook still there → kingside castling ok
        assert board.has_kingside_castling_rights(chess.WHITE)
        # Black should have both
        assert board.has_queenside_castling_rights(chess.BLACK)
        assert board.has_kingside_castling_rights(chess.BLACK)

    def test_boards_are_legal_positions(self):
        """All generated boards should be valid chess positions."""
        for pattern in get_patterns():
            for side in [chess.WHITE, chess.BLACK]:
                board = make_handicap_board(pattern, side)
                assert board.is_valid(), (
                    f"Invalid board for pattern={pattern.pattern_id} side={side}"
                )

    def test_position_to_dict(self):
        pattern = get_pattern_by_id("queen")
        pos = generate_position(pattern, chess.WHITE)
        d = pos.to_dict()
        assert d["pattern_id"] == "queen"
        assert d["handicap_side"] == "white"
        assert "fen" in d
        assert "material_diff" in d
