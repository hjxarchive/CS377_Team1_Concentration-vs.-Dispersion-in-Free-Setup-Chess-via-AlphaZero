import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.arena import evaluate_matchup_patterns
from handichess.track_a.encoding import decode_action

print("Loading trained model...")

game = ChessGame.from_matchup("rook_4pawns", "white", max_moves=400)
net = create_net_for_game(game)

checkpoint = torch.load("runs/checkpoints/final.pt", map_location="cpu")
# Check if it's a full checkpoint dict or just state dict
if "model_state_dict" in checkpoint:
    net.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    net.load_state_dict(checkpoint["state_dict"])
else:
    net.load_state_dict(checkpoint)

net.eval()

mcts_config = {
    "num_simulations": 50, # Slightly more for the real network, but still fast
    "c_puct": 1.25
}

print("Running 2 games with trained model...")
results = evaluate_matchup_patterns(
    game_class=ChessGame,
    net=net,
    pattern_ids=["rook_4pawns"],
    num_games_per_pattern=10, # 5 games per side
    mcts_config=mcts_config,
    device="cpu",
    log_path=None
)

print("\nEvaluation results:")
for pid, res in results.items():
    print(f"{pid}: {res}")

print("Done!")
