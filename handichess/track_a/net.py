"""
Residual neural network for AlphaZero.

Dual-headed architecture:
  - Shared ResNet backbone (configurable depth and width)
  - Policy head → action probabilities (softmax over action space)
  - Value head → scalar evaluation (tanh, range [-1, +1])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """A single residual block: conv → BN → ReLU → conv → BN + skip → ReLU."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-headed residual network.

    Input:  (batch, num_input_planes, board_h, board_w)
    Output: (policy_logits, value)
            policy_logits: (batch, action_size) — unnormalized log-probs
            value: (batch, 1) — scalar in [-1, +1]
    """

    def __init__(
        self,
        num_input_planes: int,
        board_size: tuple[int, int],
        action_size: int,
        num_res_blocks: int = 10,
        num_channels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.board_h, self.board_w = board_size
        self.action_size = action_size

        # Initial convolution
        self.conv_input = nn.Conv2d(
            num_input_planes, num_channels, 3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_h * self.board_w, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_h * self.board_w, 256)
        self.value_dropout = nn.Dropout(dropout)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, planes, h, w).

        Returns:
            (policy_logits, value) tuple.
        """
        # Backbone
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits (masked before softmax in MCTS)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = self.value_dropout(F.relu(self.value_fc1(v)))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(
        self,
        encoded_state: torch.Tensor,
        valid_moves: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Convenience method for single-state prediction.

        Args:
            encoded_state: (planes, h, w) tensor.
            valid_moves: Optional (action_size,) binary mask.

        Returns:
            (policy_probs, value) — policy as probabilities (after masking + softmax).
        """
        self.eval()
        with torch.no_grad():
            x = encoded_state.unsqueeze(0)
            policy_logits, value = self(x)
            policy_logits = policy_logits.squeeze(0)

            if valid_moves is not None:
                # Mask illegal moves with large negative
                policy_logits = policy_logits.masked_fill(
                    valid_moves == 0, float("-inf")
                )

            policy_probs = F.softmax(policy_logits, dim=0)
            return policy_probs, value.item()


def create_net_for_game(game, config: Optional[dict] = None) -> AlphaZeroNet:
    """
    Factory function to create a network for a specific game.

    Args:
        game: A Game instance.
        config: Optional dict with net hyperparameters.

    Returns:
        AlphaZeroNet instance.
    """
    if config is None:
        config = {}

    # Determine input planes
    board = game.get_init_board()
    canonical = game.get_canonical_form(board, 1)
    encoded = game.get_encoded_state(canonical)
    num_input_planes = encoded.shape[0]

    return AlphaZeroNet(
        num_input_planes=num_input_planes,
        board_size=game.get_board_size(),
        action_size=game.get_action_size(),
        num_res_blocks=config.get("num_res_blocks", 10),
        num_channels=config.get("num_channels", 128),
        dropout=config.get("dropout", 0.3),
    )
