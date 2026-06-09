import sys
import torch
import chess
import chess.pgn
import numpy as np

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS
from handichess.track_a.encoding import decode_action, encode_action
from handichess.track_a.baseline import RandomAgent

def play_vs_random():
    game = ChessGame()
    net_config = {"num_res_blocks": 10, "num_channels": 128}
    net = create_net_for_game(game, net_config)
    
    checkpoint = torch.load("runs/checkpoints/final.pt", map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    mcts = MCTS(game, net, num_simulations=50, c_puct=1.25, device="cpu")

    random_agent = RandomAgent(game)

    board = chess.Board()
    state = game.get_init_board()
    player = 1 # 1 for White (AlphaZero), -1 for Black (Random)

    print("--- AlphaZero (White) vs Random Agent (Black) ---")
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < 150:
        if player == 1:
            probs = mcts.search(state, player, temperature=0.0, add_noise=False)
            action = np.argmax(probs)
        else:
            action = random_agent.get_action(state, player)
            
        b = game._state_to_board(state)
        move = decode_action(action, b)
        
        # print(f"Ply {ply}: {'AlphaZero' if player == 1 else 'Random'} plays {move.uci()}")
        
        board.push(move)
        state, player = game.get_next_state(state, player, action)
        ply += 1

    result_str = board.result(claim_draw=True)
    print(f"\nGame Over! Result: {result_str} (in {ply} plies)")
    print(f"Final FEN: {board.fen()}")
    
    game_pgn = chess.pgn.Game.from_board(board)
    game_pgn.headers["White"] = "AlphaZero"
    game_pgn.headers["Black"] = "Random Agent"
    print(game_pgn)

if __name__ == "__main__":
    play_vs_random()
