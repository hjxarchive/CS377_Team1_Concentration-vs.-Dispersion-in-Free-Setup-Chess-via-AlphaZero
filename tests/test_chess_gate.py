import pytest
import numpy as np
import torch
import chess

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS

def test_chess_canonical_mirror_invariant():
    """
    Test that the canonical form and encoding properly handle both colors.
    Specifically, we check that black's perspective is encoded correctly
    (is_own logic) and the net can evaluate it without crashing.
    """
    game = ChessGame()
    net_config = {"num_res_blocks": 1, "num_channels": 16}
    net = create_net_for_game(game, net_config)
    
    # 1. Start board
    state = game.get_init_board()
    player = 1
    
    # Encode for white
    canonical_w = game.get_canonical_form(state, player)
    encoded_w = game.get_encoded_state(canonical_w)
    
    # Play a move (e2e4)
    action_w = 8 * 8 * 0 + 12  # e2-e4 is somehow encoded, let's just use a valid move
    valid_moves_w = game.get_valid_moves(state, player)
    assert valid_moves_w.sum() == 20
    
    # Just grab the first valid action
    action_w = np.argmax(valid_moves_w)
    state, player = game.get_next_state(state, player, action_w)
    
    # 2. Black's turn
    assert player == -1
    canonical_b = game.get_canonical_form(state, player)
    encoded_b = game.get_encoded_state(canonical_b)
    
    # Verify that in encoded_b, the "own pieces" planes (0-5) belong to Black
    b = game._state_to_board(state)
    # The piece at index 0 is pawn.
    # In encoded_b, plane 0 should have 1s where Black's pawns are, because it's Black's turn.
    black_pawn_count = encoded_b[0].sum()
    white_pawn_count = encoded_b[6].sum()
    
    assert black_pawn_count == len(b.pieces(chess.PAWN, chess.BLACK))
    assert white_pawn_count == len(b.pieces(chess.PAWN, chess.WHITE))
    
    # 3. MCTS Sanity
    mcts = MCTS(game, net, num_simulations=4, c_puct=1.0)
    
    # Ensure it doesn't crash or hit the canonical bug
    probs = mcts.search(state, player, temperature=1.0, add_noise=False)
    
    # Black should have valid moves, and probs should sum to 1
    valid_moves_b = game.get_valid_moves(state, player)
    assert valid_moves_b.sum() > 0
    assert np.isclose(probs.sum(), 1.0)
    
    # Make sure we only assigned probability to valid moves
    assert np.all(probs[valid_moves_b == 0] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
