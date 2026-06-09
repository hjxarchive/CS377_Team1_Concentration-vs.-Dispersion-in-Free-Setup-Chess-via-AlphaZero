import sys
import argparse
import torch
import chess
import chess.engine
import numpy as np

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.mcts import MCTS
from handichess.track_a.encoding import decode_action, encode_action

# Stockfish levels for quick sanity checks
SF_LEVELS = [
    {"name": "Skill Level 0 (~1100 Elo)", "config": {"Skill Level": 0}},
    {"name": "1320 Elo", "config": {"UCI_LimitStrength": True, "UCI_Elo": 1320}}
]

# LC0 Node limits for exact Elo calibration (BayesElo approach)
# LC0 nodes: 1 (~800), 2 (~950), 5 (~1100), 10 (~1300)
LC0_NODES = [1, 2, 4, 8, 16]

def play_one_game(engine, level_name, net, mcts, num_simulations=200):
    game = ChessGame()
    board = chess.Board()
    state = game.get_init_board()
    player = 1 # 1 for White (AlphaZero), -1 for Black (Engine)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < 150:
        if player == 1:
            probs = mcts.search(state, player, temperature=0.0, add_noise=False)
            action = np.argmax(probs)
            b = game._state_to_board(state)
            move = decode_action(action, b)
        else:
            limit = chess.engine.Limit(time=0.1)  # Time limit is a fallback, actual limit controlled by engine options
            result = engine.play(board, limit)
            move = result.move
            b = game._state_to_board(state)
            action = encode_action(move, b)
        
        board.push(move)
        state, player = game.get_next_state(state, player, action)
        ply += 1
    
    result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        score = 1.0
    elif result_str == "0-1":
        score = 0.0
    else:
        score = 0.5
        
    return level_name, result_str, score, ply

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero Elo")
    parser.add_argument("--engine", type=str, choices=["stockfish", "lc0"], default="lc0", help="Engine to test against")
    parser.add_argument("--lc0-path", type=str, default="lc0", help="Path to lc0 executable")
    parser.add_argument("--sims", type=int, default=200, help="AlphaZero MCTS simulations")
    parser.add_argument("--games", type=int, default=10, help="Games per level")
    args = parser.parse_args()

    print(f"Starting Elo evaluation gauntlet against {args.engine}...")
    
    # Load AlphaZero
    game = ChessGame()
    net_config = {"num_res_blocks": 10, "num_channels": 128}
    net = create_net_for_game(game, net_config)
    try:
        checkpoint = torch.load("runs/checkpoints/final.pt", map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print("Model checkpoint runs/checkpoints/final.pt not found.")
        sys.exit(1)
        
    net.eval()
    mcts = MCTS(game, net, num_simulations=args.sims, c_puct=1.25, device="cpu")

    results = {}
    
    if args.engine == "lc0":
        try:
            engine = chess.engine.SimpleEngine.popen_uci(args.lc0_path)
        except FileNotFoundError:
            print(f"LC0 executable not found at '{args.lc0_path}'.")
            sys.exit(1)
            
        levels = [{"name": f"LC0 Nodes={n}", "config": {"Nodes": n}} for n in LC0_NODES]
    else:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        levels = SF_LEVELS

    # Run gauntlet
    total_score = 0
    total_games = 0
    for level in levels:
        print(f"\n--- Testing against {level['name']} ---")
        engine.configure(level["config"])
        
        level_score = 0
        for g in range(args.games):
            name, res, score, ply = play_one_game(engine, level['name'], net, mcts, num_simulations=args.sims)
            print(f"  Game {g+1}: {res} (in {ply} plies)")
            level_score += score
            total_score += score
            total_games += 1
            
        print(f"  Level Total: {level_score} / {args.games}")
        results[level['name']] = level_score

    engine.quit()

    print("\n=== Final Summary ===")
    for name, score in results.items():
        win_rate = score / args.games
        print(f"{name}: {score}/{args.games} (Win Rate: {win_rate*100:.1f}%)")
        
    print("\nTo calculate BayesElo, feed these results into a standard BayesElo calculator.")
    print("Example reference: LC0 Nodes=1 is ~800, Nodes=2 is ~950, Nodes=5 is ~1100.")

if __name__ == "__main__":
    main()
