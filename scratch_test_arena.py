import sys
import chess
from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import AlphaZeroNet
from handichess.track_a.mcts import MCTS
import numpy as np

game = ChessGame.from_matchup("rook_4_pawns", "white", max_moves=180) # Using a guess for pattern_id
print(game.start_fen)

class DummyNet:
    def predict(self, state):
        return np.ones(game.get_action_size()) / game.get_action_size(), 0.0

net = DummyNet()
mcts = MCTS(game, net, num_simulations=10, c_puct=1.25, device="cpu")

state = game.get_init_board()
player = 1
move_count = 0

while move_count < 10:
    result = game.get_game_ended(state, player)
    if result != 0:
        print(f"Ended early at {move_count} with {result}")
        break
    probs = mcts.search(state, player, temperature=0.0, add_noise=False)
    action = np.argmax(probs)
    b = game._state_to_board(state)
    from handichess.track_a.encoding import decode_action
    print(f"Move {move_count}: {decode_action(action, b)}")
    state, player = game.get_next_state(state, player, action)
    move_count += 1

print(f"Result: {game.get_game_ended(state, player)}")
