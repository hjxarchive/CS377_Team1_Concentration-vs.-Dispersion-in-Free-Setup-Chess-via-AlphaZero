"""
End-to-end pipeline tests for AlphaZero core.
"""

import os
import numpy as np
import torch
import pytest

from handichess.track_a.game.tictactoe import TicTacToeGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS
from handichess.track_a.selfplay import SelfPlay
from handichess.track_a.trainer import Trainer

def play_against_random(game, net, num_games=20):
    """
    Play trained net against a random player.
    Net plays as +1 for half games, -1 for half games.
    Returns: (net_wins, net_losses, draws)
    """
    wins, losses, draws = 0, 0, 0
    mcts = MCTS(game, net, num_simulations=50, c_puct=1.5)

    for i in range(num_games):
        net_is_player_1 = (i % 2 == 0)
        state = game.get_init_board()
        player = 1
        
        while True:
            result = game.get_game_ended(state, player)
            if result != 0:
                break
                
            if (player == 1) == net_is_player_1:
                # Net's turn
                probs = mcts.search(state, player, temperature=0.0, add_noise=False)
                action = np.argmax(probs)
            else:
                # Random's turn
                valid = game.get_valid_moves(state, player)
                probs = valid / valid.sum()
                action = np.random.choice(len(probs), p=probs)
                
            state, player = game.get_next_state(state, player, action)
            
        # Convert result to net's perspective
        if abs(result) < 0.01:
            draws += 1
        else:
            # result is from `player`'s perspective (the one who is to move next)
            # Actually get_game_ended returns 1.0 if `player` won, -1.0 if opponent won
            # wait, game_ended usually returns from the perspective of `player`.
            # If game ended, the person who just moved won.
            # get_game_ended for TTT returns 1 if `player` won, -1 if opponent won.
            # But the last move was made by `-player`. So `result` is usually -1.0.
            
            # Let's just evaluate by looking at the board
            # player 1 is X (1), player -1 is O (-1)
            # So if player 1 won, it's a win for net if net_is_player_1
            # We can use game_ended(state, 1) to get result from player 1's perspective
            p1_result = game.get_game_ended(state, 1)
            
            if p1_result > 0:
                if net_is_player_1:
                    wins += 1
                else:
                    losses += 1
            elif p1_result < 0:
                if not net_is_player_1:
                    wins += 1
                else:
                    losses += 1

    return wins, losses, draws

@pytest.mark.timeout(180)
def test_ttt_end_to_end_learning(tmp_path):
    """
    Test that the complete pipeline (MCTS + Net + SelfPlay + Trainer) 
    can learn to play TicTacToe from scratch against a random player.
    """
    game = TicTacToeGame()
    
    # Tiny network config for fast TTT learning
    net_config = {
        "num_res_blocks": 2,
        "num_channels": 32,
    }
    net = create_net_for_game(game, net_config)
    
    mcts_config = {
        "num_simulations": 25,
        "c_puct": 1.5,
        "dirichlet_alpha": 1.0,
        "dirichlet_epsilon": 0.25,
    }
    
    train_config = {
        "epochs_per_iteration": 10,
        "batch_size": 32,
        "learning_rate": 0.01,
        "optimizer": "adam",
    }
    
    trainer = Trainer(net, config=train_config, checkpoint_dir=str(tmp_path))
    
    # Test random initial net
    wins_before, losses_before, _ = play_against_random(game, net, num_games=10)
    
    # Run a few iterations (3 iterations, 15 games per iteration)
    for i in range(3):
        mcts = MCTS(game, net, **mcts_config)
        selfplay = SelfPlay(game, net, mcts_config=mcts_config)
        examples = selfplay.generate_games(num_games=15)
        trainer.train(examples, iteration=i)
        
    net.eval()
    wins_after, losses_after, _ = play_against_random(game, net, num_games=20)
    
    print(f"Before training: {wins_before}W, {losses_before}L")
    print(f"After training: {wins_after}W, {losses_after}L")
    
    # The agent should ideally not lose against a random player in TTT after learning.
    # TTT is extremely simple, 3 iterations of 15 games is 45 games, sufficient to beat random.
    assert losses_after == 0, f"AlphaZero lost {losses_after} games to a random player after training!"
