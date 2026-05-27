"""
Monte Carlo Tree Search (MCTS) with PUCT exploration.

Follows the AlphaZero formulation:
  - Selection: PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  - Expansion: neural network evaluation of leaf nodes
  - Backup: value propagated up the tree (careful with sign/canonical form)
  - Root Dirichlet noise for exploration
  - Temperature-based action selection
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .game.base import Game
from .net import AlphaZeroNet


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        "game", "state", "player", "parent", "action",
        "children", "visit_count", "value_sum", "prior",
        "is_expanded", "is_terminal", "terminal_value",
    ]

    def __init__(
        self,
        game: Game,
        state: np.ndarray,
        player: int,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.game = game
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self) -> float:
        """Average value (from this node's player perspective)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    AlphaZero-style MCTS.

    Usage:
        mcts = MCTS(game, net, config)
        action_probs = mcts.search(state, player, temperature=1.0)
    """

    def __init__(
        self,
        game: Game,
        net: AlphaZeroNet,
        num_simulations: int = 800,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "cpu",
    ):
        self.game = game
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    def search(
        self,
        state: np.ndarray,
        player: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS from the given state and return action probabilities.

        Args:
            state: Current board state.
            player: Current player (+1 or -1).
            temperature: Controls exploration vs exploitation in action selection.
                         0 = argmax, 1 = proportional to visit counts.
            add_noise: Whether to add Dirichlet noise at the root.

        Returns:
            Action probability distribution, shape (action_size,).
        """
        root = MCTSNode(self.game, state, player)
        self._expand(root)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal:
                action, node = self._select_child(node)
                search_path.append(node)

            # Get value for backup
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expansion + evaluation
                value = self._expand(node)

            # Backup: propagate value up the tree
            # CRITICAL: value sign must be flipped at each level
            # because players alternate
            self._backup(search_path, value)

        # Extract action probabilities from root visit counts
        return self._get_action_probs(root, temperature)

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a leaf node: evaluate with neural network and create children.

        Returns:
            Value estimate for this node's position (from node's player perspective).
        """
        # Check terminal state
        game_result = self.game.get_game_ended(node.state, node.player)
        if game_result != 0:
            node.is_terminal = True
            # game_result is from node.player's perspective
            node.terminal_value = game_result
            node.is_expanded = True
            return game_result

        # Get canonical state for neural network
        canonical = self.game.get_canonical_form(node.state, node.player)
        encoded = self.game.get_encoded_state(canonical)
        valid_moves = self.game.get_valid_moves(node.state, node.player)

        # Neural network evaluation
        encoded_tensor = torch.FloatTensor(encoded).to(self.device)
        valid_tensor = torch.FloatTensor(valid_moves).to(self.device)
        policy_probs, value = self.net.predict(encoded_tensor, valid_tensor)
        policy_probs = policy_probs.cpu().numpy()

        # Create children for all valid moves
        for action in range(self.game.get_action_size()):
            if valid_moves[action] > 0:
                next_state, next_player = self.game.get_next_state(
                    node.state, node.player, action
                )
                child = MCTSNode(
                    game=self.game,
                    state=next_state,
                    player=next_player,
                    parent=node,
                    action=action,
                    prior=policy_probs[action],
                )
                node.children[action] = child

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select the child with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            # PUCT formula
            q = child.q_value
            # Negate q because child stores value from child's player perspective,
            # but parent wants to maximize its own value
            q = -q

            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backup(self, search_path: list[MCTSNode], value: float) -> None:
        """
        Propagate the value up the search path.

        The value is from the perspective of the leaf node's player.
        We alternate the sign at each level because players alternate.
        """
        for node in reversed(search_path):
            # Value from this node's player's perspective
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip for parent (opponent's perspective)

    def _get_action_probs(
        self, root: MCTSNode, temperature: float
    ) -> np.ndarray:
        """
        Convert root visit counts to action probabilities.

        Args:
            root: The root node after search.
            temperature: 0 = argmax, >0 = proportional to N^(1/temp).

        Returns:
            Probability distribution over actions.
        """
        action_size = self.game.get_action_size()
        counts = np.zeros(action_size, dtype=np.float64)

        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature == 0:
            # Argmax (break ties randomly)
            best_actions = np.where(counts == counts.max())[0]
            probs = np.zeros(action_size, dtype=np.float64)
            probs[np.random.choice(best_actions)] = 1.0
            return probs.astype(np.float32)

        # Temperature-scaled probabilities
        counts_temp = counts ** (1.0 / temperature)
        total = counts_temp.sum()
        if total == 0:
            # Fallback: uniform over valid moves
            valid = self.game.get_valid_moves(root.state, root.player)
            return (valid / valid.sum()).astype(np.float32)

        probs = counts_temp / total
        return probs.astype(np.float32)

    def get_root_value(self, state: np.ndarray, player: int) -> float:
        """
        Quick evaluation: run MCTS and return the root value estimate.
        Useful for position evaluation / sanity checks.
        """
        root = MCTSNode(self.game, state, player)
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.is_expanded and not node.is_terminal:
                _, node = self._select_child(node)
                search_path.append(node)

            if node.is_terminal:
                value = node.terminal_value
            else:
                value = self._expand(node)

            self._backup(search_path, value)

        return root.q_value
