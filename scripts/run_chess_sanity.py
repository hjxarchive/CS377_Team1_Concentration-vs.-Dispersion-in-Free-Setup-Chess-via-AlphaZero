#!/usr/bin/env python3
"""
Run a very small AlphaZero chess sanity check.
This ensures the pipeline (MCTS, Self-Play, Training, Encoding) works end-to-end
for chess, without encountering dimension errors or canonical bugs.
"""
import sys
import logging
import torch
import numpy as np

from handichess.track_a.game.chess_std import ChessGame
from handichess.track_a.net import create_net_for_game
from handichess.track_a.trainer import Trainer
from handichess.track_a.selfplay import SelfPlay, ReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Chess Sanity Check...")
    
    game = ChessGame()
    net_config = {"num_res_blocks": 2, "num_channels": 32}
    net = create_net_for_game(game, net_config)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        
    logger.info(f"Using device: {device}")
    
    trainer = Trainer(
        net, device=device,
        checkpoint_dir="runs/sanity_check",
        config={"epochs_per_iteration": 2, "batch_size": 16},
    )
    
    mcts_config = {"num_simulations": 10, "dirichlet_alpha": 0.3}
    selfplay = SelfPlay(game, net, mcts_config=mcts_config, device=device)
    buffer = ReplayBuffer(max_size=1000)
    
    # 1. Generate a few games
    logger.info("Generating 2 self-play games for sanity check...")
    examples = selfplay.generate_games(2)
    buffer.add(examples)
    
    assert len(examples) > 0, "No examples generated!"
    logger.info(f"Generated {len(examples)} position examples.")
    
    # 2. Train on the generated data
    logger.info("Training on the examples...")
    train_data = buffer.sample(min(len(buffer), 64))
    
    try:
        stats = trainer.train(train_data, iteration=0)
        logger.info(f"Training stats: {stats}")
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        sys.exit(1)
        
    # 3. MCTS inference check for Black
    logger.info("Checking MCTS inference for Black (ensuring no canonical bug)...")
    net.eval()
    from handichess.track_a.mcts import MCTS
    mcts = MCTS(game, net, num_simulations=10, device=device)
    
    state = game.get_init_board()
    # Play e2e4
    valid_w = game.get_valid_moves(state, 1)
    state, player = game.get_next_state(state, 1, np.argmax(valid_w))
    
    # Black's turn
    assert player == -1
    probs = mcts.search(state, player, temperature=1.0)
    
    valid_b = game.get_valid_moves(state, player)
    assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1.0"
    assert np.all(probs[valid_b == 0] == 0), "Probability assigned to invalid move"
    
    logger.info("Sanity check passed successfully! The pipeline is healthy for Chess.")

if __name__ == "__main__":
    main()
