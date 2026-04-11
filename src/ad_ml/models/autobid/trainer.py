"""Training loop for AutobidNet with online learning, replay buffer, and importance sampling."""

from __future__ import annotations

import collections
import logging
import math
import random
from pathlib import Path
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ad_ml.models.autobid.model import AutobidLoss, AutobidNet, AutobidOutput

logger = logging.getLogger(__name__)


class ReplayBufferSample(NamedTuple):
    """A single transition stored in the replay buffer."""
    campaign_features: np.ndarray
    context_features: np.ndarray
    bid_label: float
    budget_utilization: float
    weight: float  # Importance sampling weight


class PrioritizedReplayBuffer:
    """Fixed-capacity replay buffer with priority-based importance sampling.

    Rare events (high bid multipliers, unusual budget states) are up-sampled
    via priority weights to correct for distribution shift.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[ReplayBufferSample] = collections.deque(maxlen=capacity)
        self._priorities: Deque[float] = collections.deque(maxlen=capacity)

    def add(
        self,
        campaign_features: np.ndarray,
        context_features: np.ndarray,
        bid_label: float,
        budget_utilization: float = 0.5,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition to the buffer."""
        if priority is None:
            # Default priority: higher for extreme bid labels
            priority = 1.0 + abs(bid_label - 1.0)  # Distance from neutral bid
        sample = ReplayBufferSample(
            campaign_features=campaign_features.astype(np.float32),
            context_features=context_features.astype(np.float32),
            bid_label=float(bid_label),
            budget_utilization=float(budget_utilization),
            weight=float(priority),
        )
        self.buffer.append(sample)
        self._priorities.append(float(priority))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Optional[Dict[str, Tensor]]:
        """Sample a batch with importance sampling weights.

        Returns None if buffer has fewer samples than batch_size.
        """
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array(self._priorities, dtype=np.float32)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

        samples = [list(self.buffer)[i] for i in indices]
        is_weights = (1.0 / (len(self.buffer) * probs[indices])) ** 0.4
        is_weights /= is_weights.max()

        return {
            "campaign_features": torch.tensor(
                np.stack([s.campaign_features for s in samples]), device=device
            ),
            "context_features": torch.tensor(
                np.stack([s.context_features for s in samples]), device=device
            ),
            "bid_label": torch.tensor(
                [s.bid_label for s in samples], dtype=torch.float32, device=device
            ),
            "budget_utilization": torch.tensor(
                [s.budget_utilization for s in samples], dtype=torch.float32, device=device
            ),
            "is_weights": torch.tensor(is_weights, dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self.buffer)


def _cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class AutobidTrainer:
    """Trainer for AutobidNet.

    Supports:
    - Mixed-precision training
    - Gradient clipping
    - Cosine LR schedule with warmup
    - Early stopping
    - Online learning via prioritized replay buffer
    - MLflow experiment tracking
    - Checkpoint save/load
    """

    def __init__(
        self,
        model: AutobidNet,
        loss_fn: AutobidLoss,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        learning_rate: float = 5e-4,
        epochs: int = 100,
        warmup_steps: int = 2000,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        replay_buffer_size: int = 100_000,
        replay_batch_size: int = 256,
        replay_update_freq: int = 4,
        checkpoint_dir: Path = Path("checkpoints/autobid"),
        device: Optional[torch.device] = None,
        local_rank: int = 0,
        world_size: int = 1,
        mlflow_experiment: str = "autobid",
        use_amp: bool = True,
    ) -> None:
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = local_rank == 0
        self.use_amp = use_amp and torch.cuda.is_available()
        self.replay_batch_size = replay_batch_size
        self.replay_update_freq = replay_update_freq
        self.mlflow_experiment = mlflow_experiment

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank])  # type: ignore[assignment]

        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.scaler = GradScaler(enabled=self.use_amp)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs
        self.scheduler = _cosine_warmup_schedule(self.optimizer, warmup_steps, total_steps)

        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)

        self._best_val_loss = float("inf")
        self._early_stop_counter = 0
        self._early_stop_patience = early_stopping_patience

        if self.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, run_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Run full training loop."""
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        if self.is_main:
            mlflow.set_experiment(self.mlflow_experiment)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(self._get_hparams())

        try:
            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(epoch)
                val_loss = self._val_epoch()

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                if self.is_main:
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                    )
                    logger.info(
                        "Epoch %d/%d  train=%.4f  val=%.4f  replay=%d",
                        epoch, self.epochs, train_loss, val_loss, len(self.replay_buffer),
                    )

                    if val_loss < self._best_val_loss:
                        self._best_val_loss = val_loss
                        self._early_stop_counter = 0
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                    else:
                        self._early_stop_counter += 1

                    if epoch % 20 == 0:
                        self._save_checkpoint(epoch, val_loss, is_best=False)

                if self._early_stop_counter >= self._early_stop_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break
        finally:
            if self.is_main:
                mlflow.end_run()

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # Add to replay buffer
            self._populate_replay_buffer(batch)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output: AutobidOutput = self.model(
                    campaign_features=batch["campaign_features"],
                    context_features=batch["context_features"],
                    budget_utilization=batch.get("budget_utilization"),
                )
                losses = self.loss_fn(
                    output,
                    bid_labels=batch["bid_label"],
                    budget_utilization=batch.get("budget_utilization"),
                )
                loss = losses["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Replay buffer update
            if step % self.replay_update_freq == 0:
                replay_batch = self.replay_buffer.sample(self.replay_batch_size, self.device)
                if replay_batch is not None:
                    self._replay_update(replay_batch)

        return total_loss / max(num_batches, 1)

    def _replay_update(self, replay_batch: Dict[str, Tensor]) -> None:
        """One gradient step on a replay buffer sample."""
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            output = self.model(
                campaign_features=replay_batch["campaign_features"],
                context_features=replay_batch["context_features"],
                budget_utilization=replay_batch.get("budget_utilization"),
            )
            losses = self.loss_fn(
                output,
                bid_labels=replay_batch["bid_label"],
                budget_utilization=replay_batch.get("budget_utilization"),
            )
            # Weight loss by importance sampling weights
            loss = losses["total"] * replay_batch["is_weights"].mean()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=self.use_amp):
                output = self.model(
                    campaign_features=batch["campaign_features"],
                    context_features=batch["context_features"],
                    budget_utilization=batch.get("budget_utilization"),
                )
                losses = self.loss_fn(
                    output,
                    bid_labels=batch["bid_label"],
                    budget_utilization=batch.get("budget_utilization"),
                )
            total_loss += losses["total"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _populate_replay_buffer(self, batch: Dict[str, Tensor]) -> None:
        """Add current batch samples to replay buffer."""
        cf = batch["campaign_features"].cpu().numpy()
        ctx = batch["context_features"].cpu().numpy()
        labels = batch["bid_label"].cpu().numpy()
        budgets = batch.get("budget_utilization")
        budget_arr = budgets.cpu().numpy() if budgets is not None else np.full(len(labels), 0.5)

        for i in range(len(labels)):
            self.replay_buffer.add(cf[i], ctx[i], float(labels[i]), float(budget_arr[i]))

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        path = self.checkpoint_dir / ("best.pt" if is_best else f"epoch_{epoch:04d}.pt")
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> int:
        state = torch.load(path, map_location=self.device)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        return int(state["epoch"])

    def _get_hparams(self) -> Dict[str, object]:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        return {
            "input_dim": raw_model.input_dim,
            "min_bid": raw_model.min_bid,
            "max_bid": raw_model.max_bid,
            "epochs": self.epochs,
            "replay_buffer_size": self.replay_buffer.capacity,
            "use_amp": self.use_amp,
        }
