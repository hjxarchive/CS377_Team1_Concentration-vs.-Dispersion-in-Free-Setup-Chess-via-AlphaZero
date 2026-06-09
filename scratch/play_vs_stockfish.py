import torch
import chess
import chess.engine
import numpy as np
import chess.pgn

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS

def play_game():
    # 1. Initialize AlphaZero
    print("Loading AlphaZero (final.pt)...")
    game = ChessGame()
    net_config = {"num_res_blocks": 10, "num_channels": 128}
    net = create_net_for_game(game, net_config)
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load("runs/checkpoints/final.pt", map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Error loading final.pt: {e}")
        return

    net.eval()
    mcts = MCTS(game, net, num_simulations=200, c_puct=1.25, device="cpu")

    # 2. Initialize Stockfish
    print("Loading Stockfish...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1500})
    except Exception as e:
        print(f"Failed to start Stockfish: {e}")
        return

    # 3. Play Game
    board = chess.Board()
    state = game.get_init_board()
    player = 1 # 1 for White, -1 for Black

    from handichess.track_a.encoding import decode_action, encode_action

    print("\n--- Game Start: AlphaZero (White) vs Stockfish 1500 Elo (Black) ---")
    
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < 100:
        if player == 1:
            # AlphaZero Move
            probs = mcts.search(state, player, temperature=0.0, add_noise=False)
            action = np.argmax(probs)
            b = game._state_to_board(state)
            move = decode_action(action, b)
            move_str = move.uci()
            print(f"AlphaZero plays: {move_str}")
        else:
            # Stockfish Move
            limit = chess.engine.Limit(time=0.1)
            result = engine.play(board, limit)
            move = result.move
            b = game._state_to_board(state)
            action = encode_action(move, b)
            print(f"Stockfish plays: {move.uci()}")
        
        board.push(move)
        state, player = game.get_next_state(state, player, action)
        ply += 1

    engine.quit()
    
    print("\n--- Game Over ---")
    print(f"Result: {board.result(claim_draw=True)}")
    print(f"PGN: {board.fen()}") # Just print PGN or FEN
    game_pgn = chess.pgn.Game.from_board(board)
    game_pgn.headers["White"] = "AlphaZero"
    game_pgn.headers["Black"] = "Stockfish 1500"
    print(game_pgn)

if __name__ == "__main__":
    play_game()
