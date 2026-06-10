import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import AlphaZeroNet
from handichess.track_a.mcts import MCTS
from handichess.track_a.encoding import NUM_INPUT_PLANES, ACTION_PLANES, decode_action

print("Starting smoke test with move outputs...")

class DummyNet(AlphaZeroNet):
    def __init__(self):
        super().__init__(num_input_planes=NUM_INPUT_PLANES, board_size=(8,8), action_size=8*8*ACTION_PLANES)
    
    def predict(self, encoded_state, valid_moves):
        batch_size = encoded_state.shape[0]
        probs = valid_moves / (valid_moves.sum(dim=1, keepdim=True) + 1e-8)
        values = torch.rand(batch_size, 1) * 0.2 - 0.1
        return probs, values

net = DummyNet()
game = ChessGame.from_matchup("rook_4pawns", "white", max_moves=400)
mcts = MCTS(game, net, num_simulations=10, c_puct=1.25, device="cpu")

for game_idx in range(2):
    print(f"\n--- Game {game_idx + 1} ---")
    state = game.get_init_board()
    player = 1
    move_count = 0
    moves_played = []

    while move_count < 20: # Only play 20 half-moves to keep output concise
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
            
        # Decode action to UCI string
        b = game._state_to_board(state)
        uci_move = decode_action(action, b).uci()
        moves_played.append(uci_move)
        
        state, player = game.get_next_state(state, player, action)
        move_count += 1

    print(f"First 20 moves: {' '.join(moves_played)}")
    print(f"End state FEN: {game._state_to_board(state).fen()}")
    
print("\nSmoke test completed successfully!")
