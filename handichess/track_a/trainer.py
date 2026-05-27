"""
Training loop for AlphaZero.

Loss = policy_CE + value_MSE + L2_regularization.
Supports SGD with momentum or Adam, step LR decay, and AMP.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from .net import AlphaZeroNet
from .selfplay import TrainingExample

logger = logging.getLogger(__name__)


class Trainer:
    """
    AlphaZero trainer.

    Manages the training loop: takes training examples from self-play,
    trains the network, saves checkpoints.
    """

    def __init__(
        self,
        net: AlphaZeroNet,
        config: Optional[dict] = None,
        device: str = "cpu",
        checkpoint_dir: str = "runs/checkpoints",
    ):
        self.net = net
        self.device = device
        self.net.to(device)

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.net)
        else:
            self.model = self.net

        # Config
        c = config or {}
        self.epochs = c.get("epochs_per_iteration", 10)
        self.batch_size = c.get("batch_size", 256)
        self.lr = c.get("learning_rate", 0.01)
        self.weight_decay = c.get("weight_decay", 1e-4)
        self.momentum = c.get("momentum", 0.9)
        self.optimizer_type = c.get("optimizer", "sgd")
        self.use_amp = c.get("use_amp", False) and device != "cpu"
        self.lr_schedule = c.get("lr_schedule", [])
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        # AMP scaler
        self.scaler = GradScaler("cuda" if self.device == "cuda" else "cpu", enabled=self.use_amp)

        # Loss functions
        self.value_loss_fn = nn.MSELoss()

        # Tracking
        self.iteration = 0
        self.train_history: list[dict] = []

    def train(
        self,
        examples: list[TrainingExample],
        iteration: Optional[int] = None,
    ) -> dict:
        """
        Train the network on a batch of training examples.

        Args:
            examples: Training examples from self-play.
            iteration: Current training iteration (for LR scheduling).

        Returns:
            Dict with loss statistics.
        """
        if iteration is not None:
            self.iteration = iteration
            self._update_lr()

        # Prepare data
        states = np.array([e.encoded_state for e in examples])
        policies = np.array([e.policy_target for e in examples])
        values = np.array([e.value_target for e in examples], dtype=np.float32)

        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).to(self.device).unsqueeze(1)

        dataset = TensorDataset(states_t, policies_t, values_t)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        total_loss = 0.0
        total_pi_loss = 0.0
        total_v_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs):
            for batch_states, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()

                with autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.use_amp):
                    policy_logits, value_pred = self.model(batch_states)

                    # Policy loss: cross-entropy with MCTS visit distribution
                    pi_loss = -torch.mean(
                        torch.sum(batch_policies * torch.log_softmax(policy_logits, dim=1), dim=1)
                    )

                    # Value loss: MSE
                    v_loss = self.value_loss_fn(value_pred, batch_values)

                    loss = pi_loss + v_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()
                num_batches += 1

        stats = {
            "total_loss": total_loss / max(num_batches, 1),
            "policy_loss": total_pi_loss / max(num_batches, 1),
            "value_loss": total_v_loss / max(num_batches, 1),
            "num_examples": len(examples),
            "num_batches": num_batches,
            "iteration": self.iteration,
        }
        self.train_history.append(stats)

        logger.info(
            f"Iteration {self.iteration}: "
            f"loss={stats['total_loss']:.4f} "
            f"(pi={stats['policy_loss']:.4f}, v={stats['value_loss']:.4f}) "
            f"examples={len(examples)}"
        )

        return stats

    def _update_lr(self) -> None:
        """Update learning rate based on schedule."""
        for milestone, lr in self.lr_schedule:
            if self.iteration >= milestone:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.iteration:04d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save({
            "iteration": self.iteration,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
        }, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.train_history = checkpoint.get("train_history", [])
        logger.info(f"Checkpoint loaded: {filepath} (iteration {self.iteration})")

    def save_best(self) -> str:
        """Save the best model."""
        return self.save_checkpoint("best.pt")
