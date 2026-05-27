"""
Monte Carlo Tree Search (MCTS) with PUCT exploration.

Supports both unbatched `search()` for single evaluations and step-based
methods (`find_leaf`, `expand_and_backup`) for Batched Game Self-Play.
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
        Unbatched search (for Arena / single evaluations).
        """
        root = MCTSNode(self.game, state, player)
        
        # Initial expansion
        leaf_data = self.find_leaf(root)
        if leaf_data is not None:
            search_path, node, encoded, valid = leaf_data
            enc_t = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
            val_t = torch.FloatTensor(valid).unsqueeze(0).to(self.device)
            p, v = self.net.predict(enc_t, val_t)
            self.expand_and_backup(search_path, node, p[0].cpu().numpy(), v[0].item())

        if add_noise:
            self.add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            leaf_data = self.find_leaf(root)
            if leaf_data is not None:
                search_path, node, encoded, valid = leaf_data
                enc_t = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
                val_t = torch.FloatTensor(valid).unsqueeze(0).to(self.device)
                p, v = self.net.predict(enc_t, val_t)
                self.expand_and_backup(search_path, node, p[0].cpu().numpy(), v[0].item())

        return self.get_action_probs(root, temperature)

    def find_leaf(
        self, root: MCTSNode
    ) -> Optional[tuple[list[MCTSNode], MCTSNode, np.ndarray, np.ndarray]]:
        """
        Traverse the tree to find an unexpanded leaf node.
        If a terminal node is reached, it is backed up immediately and returns None.
        """
        node = root
        search_path = [node]

        while node.is_expanded and not node.is_terminal:
            _, node = self._select_child(node)
            search_path.append(node)

        if node.is_terminal:
            self.backup(search_path, node.terminal_value)
            return None

        # Unexpanded leaf: Check if it's actually terminal
        game_result = self.game.get_game_ended(node.state, node.player)
        if game_result != 0:
            node.is_terminal = True
            node.terminal_value = game_result
            node.is_expanded = True
            self.backup(search_path, game_result)
            return None

        canonical = self.game.get_canonical_form(node.state, node.player)
        encoded = self.game.get_encoded_state(canonical)
        valid_moves = self.game.get_valid_moves(node.state, node.player)

        return search_path, node, encoded, valid_moves

    def expand_and_backup(
        self, search_path: list[MCTSNode], node: MCTSNode, policy_probs: np.ndarray, value: float
    ):
        """
        Create children from neural net policy and backpropagate value.
        """
        valid_moves = self.game.get_valid_moves(node.state, node.player)

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
        self.backup(search_path, value)

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(max(1, node.visit_count))

        for action, child in node.children.items():
            q = child.q_value
            q = -q

            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, search_path: list[MCTSNode], value: float) -> None:
        """
        Propagate value up the search path.
        Flips sign at each level because players alternate.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def add_dirichlet_noise(self, root: MCTSNode):
        """Adds exploration noise to root children priors."""
        if root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

    def get_action_probs(
        self, root: MCTSNode, temperature: float
    ) -> np.ndarray:
        """
        Convert root visit counts to action probabilities.
        """
        action_size = self.game.get_action_size()
        counts = np.zeros(action_size, dtype=np.float64)

        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature == 0:
            best_actions = np.where(counts == counts.max())[0]
            probs = np.zeros(action_size, dtype=np.float64)
            probs[np.random.choice(best_actions)] = 1.0
            return probs.astype(np.float32)

        counts_temp = counts ** (1.0 / temperature)
        total = counts_temp.sum()
        if total == 0:
            valid = self.game.get_valid_moves(root.state, root.player)
            return (valid / valid.sum()).astype(np.float32)

        probs = counts_temp / total
        return probs.astype(np.float32)

    def get_root_value(self, state: np.ndarray, player: int) -> float:
        """
        Quick evaluation (useful for tests).
        """
        root = MCTSNode(self.game, state, player)
        leaf = self.find_leaf(root)
        if leaf is not None:
            _, node, enc, val = leaf
            p, v = self.net.predict(
                torch.FloatTensor(enc).unsqueeze(0).to(self.device),
                torch.FloatTensor(val).unsqueeze(0).to(self.device)
            )
            self.expand_and_backup([root], node, p[0].cpu().numpy(), v[0].item())
            
        for _ in range(self.num_simulations):
            leaf = self.find_leaf(root)
            if leaf is not None:
                search_path, node, enc, val = leaf
                p, v = self.net.predict(
                    torch.FloatTensor(enc).unsqueeze(0).to(self.device),
                    torch.FloatTensor(val).unsqueeze(0).to(self.device)
                )
                self.expand_and_backup(search_path, node, p[0].cpu().numpy(), v[0].item())

        return root.q_value
