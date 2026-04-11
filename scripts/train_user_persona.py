"""CLI script for training the User Persona model."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from ad_ml.config.settings import UserPersonaConfig
from ad_ml.data.dataset import UserBehaviorDataset, collate_user_sequences
from ad_ml.models.user_persona.model import PersonaLoss, UserPersonaNet
from ad_ml.models.user_persona.trainer import PersonaTrainer
from ad_ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the User Persona segmentation model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/user_persona_config.yaml"),
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/user_persona"),
        help="Directory containing preprocessed training data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/user_persona"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="user-persona",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed data-parallel training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> UserPersonaConfig:
    """Load YAML config and create UserPersonaConfig."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return UserPersonaConfig(**raw)


def build_datasets(
    data_dir: Path,
    config: UserPersonaConfig,
) -> tuple[UserBehaviorDataset, UserBehaviorDataset]:
    """Load preprocessed train and val datasets from data_dir.

    Expects:
        {data_dir}/train_sequences.npy     - list of seq arrays saved with np.save(allow_pickle=True)
        {data_dir}/train_user_features.npy - (N, user_feature_dim) array
        {data_dir}/val_sequences.npy
        {data_dir}/val_user_features.npy
    """
    train_seqs = list(np.load(data_dir / "train_sequences.npy", allow_pickle=True))
    train_ufeats = np.load(data_dir / "train_user_features.npy")
    val_seqs = list(np.load(data_dir / "val_sequences.npy", allow_pickle=True))
    val_ufeats = np.load(data_dir / "val_user_features.npy")

    train_ds = UserBehaviorDataset(train_seqs, train_ufeats, max_length=config.max_sequence_length)
    val_ds = UserBehaviorDataset(val_seqs, val_ufeats, max_length=config.max_sequence_length)
    return train_ds, val_ds


def train_worker(
    local_rank: int,
    world_size: int,
    args: argparse.Namespace,
    config: UserPersonaConfig,
) -> None:
    """Worker function for both single-process and distributed training."""
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=local_rank,
            world_size=world_size,
        )

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    train_ds, val_ds = build_datasets(args.data_dir, config)

    from torch.utils.data import DataLoader, DistributedSampler

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_user_sequences,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_user_sequences,
        pin_memory=device.type == "cuda",
    )

    # Infer feature dims from first batch
    sample_batch = next(iter(train_loader))
    seq_feature_dim = sample_batch["sequences"].shape[-1]
    user_feature_dim = sample_batch["user_features"].shape[-1]

    model = UserPersonaNet(
        seq_feature_dim=seq_feature_dim,
        user_feature_dim=user_feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        num_segments=config.num_segments,
        gru_hidden_size=config.gru_hidden_size,
        gru_num_layers=config.gru_num_layers,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    )

    loss_fn = PersonaLoss(
        reconstruction_weight=config.reconstruction_weight,
        clustering_weight=config.clustering_weight,
        contrastive_weight=config.contrastive_weight,
        temperature=config.contrastive_temperature,
    )

    trainer = PersonaTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        grad_clip_norm=config.grad_clip_norm,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        mlflow_experiment=args.experiment_name,
    )

    if args.resume and args.resume.exists():
        trainer.load_checkpoint(args.resume)

    history = trainer.train(run_name=args.run_name)

    if local_rank == 0:
        logger.info(
            "Training complete. Best val_loss=%.4f",
            min(history["val_loss"]),
        )

    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)
    set_seed(args.seed)

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    if args.distributed and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        logger.info("Launching distributed training on %d GPUs", world_size)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        mp.spawn(train_worker, args=(world_size, args, config), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args, config)


if __name__ == "__main__":
    main()
