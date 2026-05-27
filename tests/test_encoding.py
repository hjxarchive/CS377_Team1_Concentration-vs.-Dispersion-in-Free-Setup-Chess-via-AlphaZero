"""Tests for chess board/action encoding."""

import chess
import numpy as np
import pytest

from handichess.track_a.encoding import (
    encode_board,
    encode_action,
    decode_action,
    get_legal_move_mask,
    NUM_INPUT_PLANES,
    ACTION_SIZE,
)


class TestBoardEncoding:
    """Test board → planes encoding."""

    def test_output_shape(self):
        board = chess.Board()
        planes = encode_board(board)
        assert planes.shape == (NUM_INPUT_PLANES, 8, 8)

    def test_dtype(self):
        planes = encode_board(chess.Board())
        assert planes.dtype == np.float32

    def test_starting_position_pieces(self):
        board = chess.Board()
        planes = encode_board(board)
        # White pawns on rank 1 (index 1 in 0-indexed)
        # Plane 0 = current player's pawns (white's pawns since white to move)
        pawn_plane = planes[0]
        for file in range(8):
            assert pawn_plane[1, file] == 1.0, f"White pawn missing at file {file}"
        # No pawns on rank 0 or ranks 2-6
        assert pawn_plane[0, :].sum() == 0
        assert pawn_plane[2:7, :].sum() == 0

    def test_color_plane(self):
        # White to move → color plane = 1
        board = chess.Board()
        planes = encode_board(board)
        assert planes[14].sum() == 64.0  # all 1s

        # Black to move → color plane = 0
        board.push(chess.Move.from_uci("e2e4"))
        planes = encode_board(board)
        assert planes[14].sum() == 0.0

    def test_castling_planes(self):
        board = chess.Board()
        planes = encode_board(board)
        # Standard position: all castling rights
        assert planes[16].sum() == 64.0  # White K
        assert planes[17].sum() == 64.0  # White Q
        assert planes[18].sum() == 64.0  # Black k
        assert planes[19].sum() == 64.0  # Black q


class TestActionEncoding:
    """Test action encode/decode roundtrip."""

    def test_roundtrip_all_legal_moves_starting(self):
        """All legal moves from starting position should roundtrip."""
        board = chess.Board()
        for move in board.legal_moves:
            action = encode_action(move, board)
            decoded = decode_action(action, board)
            assert decoded == move, (
                f"Roundtrip failed: {move} → action {action} → {decoded}"
            )

    def test_roundtrip_with_captures(self):
        """Test roundtrip with a position containing captures."""
        board = chess.Board("r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")
        for move in board.legal_moves:
            action = encode_action(move, board)
            decoded = decode_action(action, board)
            assert decoded == move, (
                f"Roundtrip failed: {move} → action {action} → {decoded}"
            )

    def test_action_range(self):
        """All encoded actions should be in [0, ACTION_SIZE)."""
        board = chess.Board()
        for move in board.legal_moves:
            action = encode_action(move, board)
            assert 0 <= action < ACTION_SIZE, (
                f"Action {action} out of range for move {move}"
            )

    def test_knight_moves(self):
        """Knight moves should use planes 56-63."""
        board = chess.Board()
        knight_moves = [m for m in board.legal_moves
                       if board.piece_type_at(m.from_square) == chess.KNIGHT]
        for move in knight_moves:
            action = encode_action(move, board)
            plane = action % 73
            assert 56 <= plane < 64, (
                f"Knight move {move} got plane {plane}, expected 56-63"
            )


class TestLegalMoveMask:
    """Test legal move mask generation."""

    def test_starting_position_count(self):
        board = chess.Board()
        mask = get_legal_move_mask(board)
        assert mask.shape == (ACTION_SIZE,)
        # Standard starting position has 20 legal moves
        assert mask.sum() == 20, f"Expected 20 legal moves, got {int(mask.sum())}"

    def test_mask_is_binary(self):
        board = chess.Board()
        mask = get_legal_move_mask(board)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_mask_matches_legal_moves(self):
        """Mask should have exactly as many 1s as there are legal moves."""
        # Test several positions
        fens = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ]
        for fen in fens:
            board = chess.Board(fen)
            mask = get_legal_move_mask(board)
            expected = len(list(board.legal_moves))
            actual = int(mask.sum())
            assert actual == expected, (
                f"FEN {fen}: expected {expected} legal moves, mask has {actual}"
            )
