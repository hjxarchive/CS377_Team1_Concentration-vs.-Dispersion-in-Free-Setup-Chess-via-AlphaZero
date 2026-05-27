"""Tests for chess game implementation (perft, legal moves, game flow)."""

import chess
import numpy as np
import pytest

from handichess.track_a.game.chess_std import ChessGame


class TestChessGameBasic:
    """Basic ChessGame functionality."""

    def test_init_standard(self):
        game = ChessGame()
        board = game.get_init_board()
        assert board is not None

    def test_board_size(self):
        game = ChessGame()
        assert game.get_board_size() == (8, 8)

    def test_action_size(self):
        game = ChessGame()
        assert game.get_action_size() == 4672

    def test_valid_moves_starting(self):
        game = ChessGame()
        board = game.get_init_board()
        valid = game.get_valid_moves(board, 1)
        # Standard starting position has 20 legal moves
        assert int(valid.sum()) == 20

    def test_game_not_ended_at_start(self):
        game = ChessGame()
        board = game.get_init_board()
        result = game.get_game_ended(board, 1)
        assert result == 0.0

    def test_string_representation(self):
        game = ChessGame()
        board = game.get_init_board()
        s = game.string_representation(board)
        assert isinstance(s, str)
        assert len(s) > 0


class TestChessGameMoves:
    """Test move application and game flow."""

    def test_make_move(self):
        game = ChessGame()
        board = game.get_init_board()

        # Find e2e4 among valid moves
        valid = game.get_valid_moves(board, 1)
        valid_actions = np.where(valid > 0)[0]

        # Apply first valid move
        action = valid_actions[0]
        new_board, next_player = game.get_next_state(board, 1, action)
        assert next_player == -1  # Black's turn

    def test_scholars_mate(self):
        """
        Test Scholar's Mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#
        """
        game = ChessGame()
        board_state = game.get_init_board()
        player = 1

        moves_uci = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        b = chess.Board()

        for uci in moves_uci:
            move = chess.Move.from_uci(uci)
            from handichess.track_a.encoding import encode_action
            action = encode_action(move, b)
            board_state, player = game.get_next_state(board_state, player, action)
            b.push(move)

        # After Qxf7#, black is checkmated
        result = game.get_game_ended(board_state, player)
        # player is now -1 (black), and black lost → result should be -1
        # (from black's perspective, they lost)
        assert result != 0, "Game should be over after Scholar's Mate"


class TestChessHandicap:
    """Test handicap starting positions."""

    def test_from_handicap_queen(self):
        game = ChessGame.from_handicap("queen", "white")
        board = game.get_init_board()

        # Should be valid and game not over
        result = game.get_game_ended(board, 1)
        assert result == 0.0

        # Should have valid moves
        valid = game.get_valid_moves(board, 1)
        assert valid.sum() > 0

    def test_from_handicap_black(self):
        game = ChessGame.from_handicap("queen", "black")
        board = game.get_init_board()
        result = game.get_game_ended(board, 1)
        assert result == 0.0

    def test_handicap_different_from_standard(self):
        standard = ChessGame()
        handicap = ChessGame.from_handicap("queen", "white")

        s1 = standard.string_representation(standard.get_init_board())
        s2 = handicap.string_representation(handicap.get_init_board())
        assert s1 != s2, "Handicap position should differ from standard"


class TestPerft:
    """
    Perft tests: verify move generation correctness by counting
    the number of possible positions at various depths.

    Reference values from https://www.chessprogramming.org/Perft_Results
    """

    def _perft(self, board: chess.Board, depth: int) -> int:
        """Count nodes at given depth."""
        if depth == 0:
            return 1
        nodes = 0
        for move in board.legal_moves:
            board.push(move)
            nodes += self._perft(board, depth - 1)
            board.pop()
        return nodes

    def test_perft_depth1(self):
        """Starting position, depth 1 = 20 moves."""
        board = chess.Board()
        assert self._perft(board, 1) == 20

    def test_perft_depth2(self):
        """Starting position, depth 2 = 400 positions."""
        board = chess.Board()
        assert self._perft(board, 2) == 400

    def test_perft_depth3(self):
        """Starting position, depth 3 = 8902 positions."""
        board = chess.Board()
        assert self._perft(board, 3) == 8902

    def test_perft_handicap_depth1(self):
        """Handicap positions should have valid move counts."""
        from handichess.common.handicap import get_pattern_by_id, make_handicap_board

        pattern = get_pattern_by_id("queen")
        board = make_handicap_board(pattern, chess.WHITE)

        nodes = self._perft(board, 1)
        # Without queen, white has different move count than standard
        assert nodes > 0
        # Just verify it's a valid count (queen removal opens d1 for king)
        standard_nodes = self._perft(chess.Board(), 1)
        assert nodes != standard_nodes, "Handicap should differ from standard"
