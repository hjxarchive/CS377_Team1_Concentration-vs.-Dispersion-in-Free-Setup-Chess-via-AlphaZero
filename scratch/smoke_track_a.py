import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import AlphaZeroNet
from handichess.track_a.arena import evaluate_matchup_patterns
from handichess.track_a.encoding import NUM_INPUT_PLANES, ACTION_PLANES
from handichess.common.gamelog import GameLog

print("Running Track A Smoke Test...")

class DummyNet(AlphaZeroNet):
    def __init__(self):
        super().__init__(num_input_planes=NUM_INPUT_PLANES, board_size=(8,8), action_size=8*8*ACTION_PLANES)
    def predict(self, encoded_state, valid_moves):
        probs = valid_moves / (valid_moves.sum(dim=1, keepdim=True) + 1e-8)
        return probs, torch.zeros(encoded_state.shape[0], 1)

log_path = "scratch/track_a_log.jsonl"
if os.path.exists(log_path):
    os.remove(log_path)

evaluate_matchup_patterns(
    game_class=ChessGame,
    net=DummyNet(),
    pattern_ids=["rook_4pawns"],
    num_games_per_pattern=2,
    mcts_config={"num_simulations": 10},
    device="cpu",
    log_path=log_path
)

print(f"\nChecking JSONL Log at {log_path}:")
with open(log_path, 'r') as f:
    for line in f:
        print(line.strip())
