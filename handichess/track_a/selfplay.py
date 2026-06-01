"""
Self-play pipeline for AlphaZero training.

Generates training data by having the neural network play against itself.
Supports Batched MCTS evaluations to run hundreds of games concurrently
and saturate multi-GPU setups.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .game.base import Game
from .net import AlphaZeroNet
from .mcts import MCTS, MCTSNode

logger = logging.getLogger(__name__)


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

        # Set eval mode once (avoid redundant calls per batch)
        self.model.eval()

        move_step = 0
        total_gpu_time = 0.0
        total_cpu_time = 0.0
        total_gpu_calls = 0
        finished_games = 0
        selfplay_start = time.time()

        logger.info(f"  [selfplay] Starting {num_games} games | sims={self.num_simulations} | max_moves={self.max_moves} | device={self.device}")

        # Main self-play loop
        while active_games:
            move_step_start = time.time()

            # 1. Expand all roots
            t0 = time.time()
            batch_data = []
            for g in active_games:
                res = g.mcts.find_leaf(g.root)
                if res is not None:
                    batch_data.append((g, res))
            cpu_find_time = time.time() - t0

            if batch_data:
                t0 = time.time()
                enc_batch = torch.FloatTensor(np.array([item[1][2] for item in batch_data])).to(self.device)
                val_batch = torch.FloatTensor(np.array([item[1][3] for item in batch_data])).to(self.device)
                
                # Inference
                p_batch, v_batch = self._predict_batch(enc_batch, val_batch)
                p_batch = p_batch.cpu().numpy()
                v_batch = v_batch.cpu().numpy()
                gpu_time = time.time() - t0
                total_gpu_time += gpu_time
                total_gpu_calls += 1

                for i, (g, res) in enumerate(batch_data):
                    search_path, node, _, valid = res
                    g.mcts.expand_and_backup(search_path, node, p_batch[i], float(v_batch[i][0]), valid)

            # 2. Add Dirichlet Noise
            for g in active_games:
                g.mcts.add_dirichlet_noise(g.root)

            # 3. Run MCTS Simulations concurrently
            sim_cpu_time = 0.0
            sim_gpu_time = 0.0
            sim_gpu_calls = 0
            sim_total_batch = 0

            for _ in range(self.num_simulations):
                t0 = time.time()
                batch_data = []
                for g in active_games:
                    res = g.mcts.find_leaf(g.root)
                    if res is not None:
                        batch_data.append((g, res))
                sim_cpu_time += time.time() - t0

                if batch_data:
                    t0 = time.time()
                    enc_batch = torch.FloatTensor(np.array([item[1][2] for item in batch_data])).to(self.device)
                    val_batch = torch.FloatTensor(np.array([item[1][3] for item in batch_data])).to(self.device)
                    
                    p_batch, v_batch = self._predict_batch(enc_batch, val_batch)
                    p_batch = p_batch.cpu().numpy()
                    v_batch = v_batch.cpu().numpy()
                    dt = time.time() - t0
                    sim_gpu_time += dt
                    sim_gpu_calls += 1
                    sim_total_batch += len(batch_data)

                    for i, (g, res) in enumerate(batch_data):
                        search_path, node, _, valid = res
                        g.mcts.expand_and_backup(search_path, node, p_batch[i], float(v_batch[i][0]), valid)

            total_gpu_time += sim_gpu_time
            total_cpu_time += sim_cpu_time
            total_gpu_calls += sim_gpu_calls

            # 4. Advance states and check for termination
            next_active = []
            step_finished = 0
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
                    step_finished += 1
                    finished_games += 1
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
            move_step += 1
            move_step_elapsed = time.time() - move_step_start

            # Log every 10 move steps
            if move_step % 10 == 0 or not active_games:
                avg_batch = sim_total_batch / max(1, sim_gpu_calls)
                gpu_pct = sim_gpu_time / max(1e-9, sim_cpu_time + sim_gpu_time) * 100
                elapsed = time.time() - selfplay_start
                logger.info(
                    f"  [selfplay] step={move_step:3d} | "
                    f"active={len(active_games):3d} finished={finished_games:3d} | "
                    f"step_time={move_step_elapsed:.1f}s | "
                    f"sim: cpu={sim_cpu_time:.2f}s gpu={sim_gpu_time:.3f}s batch={avg_batch:.0f} gpu%={gpu_pct:.1f}% | "
                    f"elapsed={elapsed:.0f}s"
                )

        total_elapsed = time.time() - selfplay_start
        gpu_pct_total = total_gpu_time / max(1e-9, total_cpu_time + total_gpu_time) * 100
        logger.info(
            f"  [selfplay] DONE | {finished_games} games in {total_elapsed:.1f}s | "
            f"examples={len(all_examples)} | "
            f"gpu_calls={total_gpu_calls} total_gpu={total_gpu_time:.2f}s total_cpu={total_cpu_time:.2f}s gpu%={gpu_pct_total:.1f}%"
        )

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
