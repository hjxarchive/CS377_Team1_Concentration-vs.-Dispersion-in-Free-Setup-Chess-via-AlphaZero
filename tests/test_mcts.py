"""Tests for MCTS implementation."""

import numpy as np
import torch
import pytest

from handichess.track_a.game.tictactoe import TicTacToeGame
from handichess.track_a.net import AlphaZeroNet
from handichess.track_a.mcts import MCTS, MCTSNode


class TestMCTSBasic:
    """Basic MCTS functionality tests."""

    @pytest.fixture
    def ttt_setup(self):
        """Set up TicTacToe game with a small network."""
        game = TicTacToeGame()
        net = AlphaZeroNet(
            num_input_planes=3,
            board_size=(3, 3),
            action_size=9,
            num_res_blocks=2,
            num_channels=32,
        )
        return game, net

    def test_search_returns_valid_distribution(self, ttt_setup):
        game, net = ttt_setup
        mcts = MCTS(game, net, num_simulations=50)
        state = game.get_init_board()
        probs = mcts.search(state, player=1, temperature=1.0)

        assert probs.shape == (9,)
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probs sum to {probs.sum()}"
        assert (probs >= 0).all(), "Negative probabilities found"

    def test_search_only_valid_moves(self, ttt_setup):
        """MCTS should only assign probability to valid moves."""
        game, net = ttt_setup
        mcts = MCTS(game, net, num_simulations=50)
        state = game.get_init_board()
        valid = game.get_valid_moves(state, 1)
        probs = mcts.search(state, player=1, temperature=1.0)

        for i in range(9):
            if valid[i] == 0:
                assert probs[i] == 0.0, (
                    f"Invalid move {i} has prob {probs[i]}"
                )

    def test_search_temperature_zero(self, ttt_setup):
        """Temperature 0 should produce a one-hot distribution."""
        game, net = ttt_setup
        mcts = MCTS(game, net, num_simulations=50)
        state = game.get_init_board()
        probs = mcts.search(state, player=1, temperature=0.0, add_noise=False)

        assert abs(probs.max() - 1.0) < 1e-5
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_search_with_partial_board(self, ttt_setup):
        """MCTS should work on a partially-filled board."""
        game, net = ttt_setup
        mcts = MCTS(game, net, num_simulations=50)

        # X plays center
        state = game.get_init_board()
        state, player = game.get_next_state(state, 1, 4)
        # O plays corner
        state, player = game.get_next_state(state, player, 0)

        probs = mcts.search(state, player=player, temperature=1.0)
        valid = game.get_valid_moves(state, player)

        # Should only have 7 valid moves
        assert int(valid.sum()) == 7
        assert abs(probs.sum() - 1.0) < 1e-5


class TestMCTSSanity:
    """Sanity checks for MCTS behavior."""

    def test_uniform_prior_winning_move(self):
        """
        With a uniform-prior net and enough simulations,
        MCTS should find a winning move in an obvious position.
        """
        game = TicTacToeGame()

        # Create a "uniform" net by using random weights
        net = AlphaZeroNet(
            num_input_planes=3,
            board_size=(3, 3),
            action_size=9,
            num_res_blocks=1,
            num_channels=16,
        )

        mcts = MCTS(game, net, num_simulations=200)

        # Set up a position where X can win in one move:
        # X X .     (X plays position 2 to win)
        # O O .
        # . . .
        state = np.zeros((3, 3), dtype=np.float32)
        state[0, 0] = 1   # X
        state[0, 1] = 1   # X
        state[1, 0] = -1  # O
        state[1, 1] = -1  # O

        probs = mcts.search(state, player=1, temperature=0.0, add_noise=False)
        best_action = np.argmax(probs)

        # Action 2 = (0,2) = winning move for X
        assert best_action == 2, (
            f"Expected action 2 (winning move), got {best_action}"
        )

    def test_blocks_opponent_win(self):
        """MCTS should block an opponent's winning threat."""
        game = TicTacToeGame()
        net = AlphaZeroNet(
            num_input_planes=3,
            board_size=(3, 3),
            action_size=9,
            num_res_blocks=1,
            num_channels=16,
        )
        mcts = MCTS(game, net, num_simulations=200)

        # O O .    (X must play position 2 to block O)
        # X . .
        # . . .
        state = np.zeros((3, 3), dtype=np.float32)
        state[0, 0] = -1  # O
        state[0, 1] = -1  # O
        state[1, 0] = 1   # X

        # Canonical form for player X (+1) — already canonical
        probs = mcts.search(state, player=1, temperature=0.0, add_noise=False)
        best_action = np.argmax(probs)

        # Action 2 = (0,2) = blocking move
        assert best_action == 2, (
            f"Expected action 2 (blocking move), got {best_action}"
        )
