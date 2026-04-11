"""Training loop for UserPersonaNet with mixed precision, DDP, and early stopping."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from ad_ml.models.user_persona.model import PersonaLoss, PersonaOutput, UserPersonaNet

logger = logging.getLogger(__name__)


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_fraction: float = 0.0,
) -> LambdaLR:
    """Cosine annealing LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_fraction, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Track validation loss and signal when training should stop."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class PersonaTrainer:
    """Trainer for UserPersonaNet.

    Supports:
    - Mixed-precision training (torch.cuda.amp)
    - Gradient clipping
    - Cosine annealing with linear warmup
    - Early stopping
    - Distributed data-parallel training
    - MLflow experiment logging
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model: UserPersonaNet,
        loss_fn: PersonaLoss,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        learning_rate: float = 1e-3,
        epochs: int = 50,
        warmup_steps: int = 1000,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 5,
        checkpoint_dir: Path = Path("checkpoints/user_persona"),
        device: Optional[torch.device] = None,
        local_rank: int = 0,
        world_size: int = 1,
        mlflow_experiment: str = "user-persona",
        use_amp: bool = True,
    ) -> None:
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = local_rank == 0
        self.use_amp = use_amp and torch.cuda.is_available()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Wrap with DDP if distributed
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank])  # type: ignore[assignment]

        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs
        self.scheduler = _cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.mlflow_experiment = mlflow_experiment
        if self.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, run_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns:
            History dict with keys 'train_loss' and 'val_loss'.
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        if self.is_main:
            mlflow.set_experiment(self.mlflow_experiment)
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(self._get_hparams())

        best_val_loss = float("inf")

        try:
            for epoch in range(1, self.epochs + 1):
                if self.world_size > 1:
                    sampler = self.train_loader.sampler
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(epoch)

                train_loss = self._train_epoch(epoch)
                val_loss = self._val_epoch(epoch)

                if self.world_size > 1:
                    # Aggregate losses across ranks
                    train_tensor = torch.tensor(train_loss, device=self.device)
                    val_tensor = torch.tensor(val_loss, device=self.device)
                    dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
                    train_loss = float(train_tensor.item())
                    val_loss = float(val_tensor.item())

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                if self.is_main:
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                    )
                    logger.info(
                        "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                        epoch, self.epochs, train_loss, val_loss,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(epoch, val_loss, is_best=True)

                    if epoch % 10 == 0:
                        self._save_checkpoint(epoch, val_loss, is_best=False)

                if self.early_stopping.step(val_loss):
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

        for batch in self.train_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output: PersonaOutput = self.model(
                    sequences=batch["sequences"],
                    user_features=batch["user_features"],
                    seq_lengths=batch["seq_lengths"],
                    attention_mask=batch.get("attention_mask"),
                )
                # Target: mean of input sequence features
                mask = batch.get("attention_mask")
                if mask is not None:
                    seq_sum = (batch["sequences"] * mask.unsqueeze(-1).float()).sum(1)
                    seq_mean = seq_sum / batch["seq_lengths"].unsqueeze(-1).float().clamp(min=1)
                else:
                    seq_mean = batch["sequences"].mean(1)

                losses = self.loss_fn(output, seq_mean)
                loss = losses["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(enabled=self.use_amp):
                output = self.model(
                    sequences=batch["sequences"],
                    user_features=batch["user_features"],
                    seq_lengths=batch["seq_lengths"],
                    attention_mask=batch.get("attention_mask"),
                )
                mask = batch.get("attention_mask")
                if mask is not None:
                    seq_sum = (batch["sequences"] * mask.unsqueeze(-1).float()).sum(1)
                    seq_mean = seq_sum / batch["seq_lengths"].unsqueeze(-1).float().clamp(min=1)
                else:
                    seq_mean = batch["sequences"].mean(1)

                losses = self.loss_fn(output, seq_mean)

            total_loss += losses["total"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

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
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return the epoch number."""
        state = torch.load(path, map_location=self.device)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, state["epoch"])
        return int(state["epoch"])

    def _get_hparams(self) -> Dict[str, object]:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        return {
            "embedding_dim": raw_model.embedding_dim,
            "num_segments": raw_model.num_segments,
            "epochs": self.epochs,
            "grad_clip_norm": self.grad_clip_norm,
            "world_size": self.world_size,
            "use_amp": self.use_amp,
        }
