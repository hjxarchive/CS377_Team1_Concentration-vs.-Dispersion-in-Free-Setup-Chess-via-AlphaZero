import sys
import os
import torch
import numpy as np
import chess.pgn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS
from handichess.track_a.encoding import decode_action

print("Loading trained model...")

game = ChessGame.from_matchup("rook_4pawns", "white", max_moves=400)
net = create_net_for_game(game)

checkpoint = torch.load("runs/checkpoints/final.pt", map_location="cpu")
if "model_state_dict" in checkpoint:
    net.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    net.load_state_dict(checkpoint["state_dict"])
else:
    net.load_state_dict(checkpoint)

net.eval()
mcts = MCTS(game, net, num_simulations=50, c_puct=1.25, device="cpu")

print(f"\nInitial Setup FEN: {game.start_fen}")

for game_idx in range(2):
    print(f"\n==============================")
    print(f"=== Game {game_idx + 1} ===")
    print(f"==============================\n")
    
    state = game.get_init_board()
    player = 1
    move_count = 0
    moves_played = []
    
    # To save PGN
    init_board = game._state_to_board(state)
    pgn_game = chess.pgn.Game()
    pgn_game.setup(init_board)
    pgn_game.headers["Event"] = f"Smoke Test with Trained Model (Game {game_idx+1})"
    pgn_game.headers["White"] = "AlphaZero Trained (NoQ)"
    pgn_game.headers["Black"] = "AlphaZero Trained (-Rook, -4Pawns)"
    
    pgn_node = pgn_game

    while True:
        result = game.get_game_ended(state, player)
        if result != 0:
            break

        temp = 1.0 if move_count < 10 else 0.0
        add_n = True if move_count < 10 else False
        probs = mcts.search(state, player, temperature=temp, add_noise=add_n)
        
        if temp > 0:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)
            
        b = game._state_to_board(state)
        move = decode_action(action, b)
        
        # Add to PGN
        pgn_node = pgn_node.add_variation(move)
        
        state, player = game.get_next_state(state, player, action)
        move_count += 1
        
    print(pgn_game)
    print(f"\nResult Score (from White's perspective): {result}")
