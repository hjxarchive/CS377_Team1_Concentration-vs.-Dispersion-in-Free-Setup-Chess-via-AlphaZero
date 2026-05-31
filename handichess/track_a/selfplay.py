"""
Self-play pipeline for AlphaZero training.

Generates training data by having the neural network play against itself.
Supports Batched MCTS evaluations to run hundreds of games concurrently
and saturate multi-GPU setups.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .game.base import Game
from .net import AlphaZeroNet
from .mcts import MCTS, MCTSNode


@dataclass
class TrainingExample:
    """A single training example from self-play."""
    encoded_state: np.ndarray   # (planes, h, w) — canonical form
    policy_target: np.ndarray   # (action_size,) — MCTS visit probabilities
    value_target: float         # Game outcome from this state's perspective: +1, 0, -1


class ReplayBuffer:
    """Fixed-size replay buffer for training examples."""

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, examples: list[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class ActiveGame:
    """State wrapper for a concurrently running self-play game."""
    def __init__(self, game: Game, mcts: MCTS, start_state: np.ndarray):
        self.game = game
        self.mcts = mcts
        self.state = start_state
        self.player = 1
        self.move_count = 0
        self.root = MCTSNode(game, start_state, 1)
        self.trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []


class SelfPlay:
    """
    Batched Self-play game generator.

    Plays games concurrently to batch leaf evaluations together, dramatically
    increasing GPU utilization.
    """

    def __init__(
        self,
        game: Game,
        net: AlphaZeroNet,
        mcts_config: Optional[dict] = None,
        selfplay_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.game = game
        self.net = net
        self.device = device

        # MCTS config
        mc = mcts_config or {}
        self.num_simulations = mc.get("num_simulations", 800)
        self.c_puct = mc.get("c_puct", 1.25)
        self.dirichlet_alpha = mc.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = mc.get("dirichlet_epsilon", 0.25)

        # Self-play config
        sc = selfplay_config or {}
        self.max_moves = sc.get("max_moves", 512)
        self.exploration_plies = sc.get("exploration_plies", 16)

        # Multi-GPU Support
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.net)
        else:
            self.model = self.net

    def _predict_batch(self, enc_batch: torch.Tensor, val_batch: torch.Tensor):
        """Runs batched inference, utilizing DataParallel if available."""
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(enc_batch)
            policy_logits = policy_logits.masked_fill(val_batch == 0, float("-inf"))
            policy_probs = torch.nn.functional.softmax(policy_logits, dim=1)
            return policy_probs, value

    def generate_games(
        self,
        num_games: int,
        start_states: Optional[list[np.ndarray]] = None,
    ) -> list[TrainingExample]:
        """
        Play N games concurrently and collect all training examples.
        """
        all_examples = []
        active_games = []

        # Initialize games
        for _ in range(num_games):
            if start_states:
                start = random.choice(start_states)
            else:
                start = self.game.get_init_board()

            # Each game needs its own MCTS instance for configuration parameters, 
            # although they share the same neural net.
            mcts = MCTS(
                game=self.game,
                net=self.net,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                device=self.device,
            )
            active_games.append(ActiveGame(self.game, mcts, start))

        # Main self-play loop
        while active_games:
            # 1. Expand all roots
            batch_data = []
            for g in active_games:
                res = g.mcts.find_leaf(g.root)
                if res is not None:
                    batch_data.append((g, res))

            if batch_data:
                enc_batch = torch.FloatTensor(np.array([item[1][2] for item in batch_data])).to(self.device)
                val_batch = torch.FloatTensor(np.array([item[1][3] for item in batch_data])).to(self.device)
                
                # Inference
                p_batch, v_batch = self._predict_batch(enc_batch, val_batch)
                p_batch = p_batch.cpu().numpy()
                v_batch = v_batch.cpu().numpy()

                for i, (g, res) in enumerate(batch_data):
                    search_path, node, _, valid = res
                    g.mcts.expand_and_backup(search_path, node, p_batch[i], float(v_batch[i][0]), valid)

            # 2. Add Dirichlet Noise
            for g in active_games:
                g.mcts.add_dirichlet_noise(g.root)

            # 3. Run MCTS Simulations concurrently
            for _ in range(self.num_simulations):
                batch_data = []
                for g in active_games:
                    res = g.mcts.find_leaf(g.root)
                    if res is not None:
                        batch_data.append((g, res))

                if batch_data:
                    enc_batch = torch.FloatTensor(np.array([item[1][2] for item in batch_data])).to(self.device)
                    val_batch = torch.FloatTensor(np.array([item[1][3] for item in batch_data])).to(self.device)
                    
                    p_batch, v_batch = self._predict_batch(enc_batch, val_batch)
                    p_batch = p_batch.cpu().numpy()
                    v_batch = v_batch.cpu().numpy()

                    for i, (g, res) in enumerate(batch_data):
                        search_path, node, _, valid = res
                        g.mcts.expand_and_backup(search_path, node, p_batch[i], float(v_batch[i][0]), valid)

            # 4. Advance states and check for termination
            next_active = []
            for g in active_games:
                if g.move_count < self.exploration_plies:
                    temperature = 1.0
                else:
                    temperature = 0.1

                action_probs = g.mcts.get_action_probs(g.root, temperature)

                # Store trajectory
                canonical = self.game.get_canonical_form(g.state, g.player)
                encoded = self.game.get_encoded_state(canonical)
                symmetries = self.game.get_symmetries(canonical, action_probs)
                for sym_board, sym_pi in symmetries:
                    sym_encoded = self.game.get_encoded_state(sym_board)
                    g.trajectory.append((sym_encoded, sym_pi, g.player))

                # Play move
                action = np.random.choice(len(action_probs), p=action_probs)
                g.state, g.player = self.game.get_next_state(g.state, g.player, action)
                g.move_count += 1

                # Check game end
                result = self.game.get_game_ended(g.state, g.player)
                if result == 0 and g.move_count >= self.max_moves:
                    result = 1e-4  # Draw by max moves limit

                if result != 0:
                    # Game finished! Process examples
                    for encoded, policy, p in g.trajectory:
                        if abs(result) < 0.01:
                            value = 0.0
                        elif p == g.player:
                            value = result
                        else:
                            value = -result
                        
                        all_examples.append(TrainingExample(
                            encoded_state=encoded,
                            policy_target=policy,
                            value_target=value,
                        ))
                else:
                    # Game continues
                    g.root = MCTSNode(self.game, g.state, g.player)
                    next_active.append(g)

            active_games = next_active

        return all_examples


def create_matchup_start_states(
    game_class,
    pattern_ids: list[str],
    max_moves: int = 180,
) -> list[np.ndarray]:
    """
    Create a list of starting states from match-up positions.
    """
    states = []
    for pid in pattern_ids:
        for noq_color in ["white", "black"]:
            g = game_class.from_matchup(pid, noq_color, max_moves)
            states.append(g.get_init_board())
    return states
