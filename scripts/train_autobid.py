"""CLI script for training the Autobidding model."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from ad_ml.config.settings import AutobidConfig
from ad_ml.data.dataset import CampaignBidDataset, collate_campaign_bids
from ad_ml.models.autobid.model import AutobidLoss, AutobidNet
from ad_ml.models.autobid.trainer import AutobidTrainer
from ad_ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Autobidding model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/autobid_config.yaml"),
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/autobid"),
        help="Directory containing preprocessed training data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/autobid"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default="autobid")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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


def load_config(config_path: Path) -> AutobidConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return AutobidConfig(**raw)


def build_datasets(
    data_dir: Path,
) -> tuple[CampaignBidDataset, CampaignBidDataset]:
    """Load preprocessed train and val datasets.

    Expects:
        {data_dir}/train_campaign_features.npy
        {data_dir}/train_context_features.npy
        {data_dir}/train_bid_labels.npy
        {data_dir}/train_budget_utils.npy   (optional)
        {data_dir}/val_campaign_features.npy
        {data_dir}/val_context_features.npy
        {data_dir}/val_bid_labels.npy
        {data_dir}/val_budget_utils.npy     (optional)
    """
    def _load(split: str) -> CampaignBidDataset:
        cf = np.load(data_dir / f"{split}_campaign_features.npy")
        ctx = np.load(data_dir / f"{split}_context_features.npy")
        labels = np.load(data_dir / f"{split}_bid_labels.npy")
        budget_path = data_dir / f"{split}_budget_utils.npy"
        budget = np.load(budget_path) if budget_path.exists() else None
        return CampaignBidDataset(cf, ctx, labels, budget)

    return _load("train"), _load("val")


def train_worker(
    local_rank: int,
    world_size: int,
    args: argparse.Namespace,
    config: AutobidConfig,
) -> None:
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_datasets(args.data_dir)

    from torch.utils.data import DataLoader, DistributedSampler

    if world_size > 1:
        train_sampler: object = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,  # type: ignore[arg-type]
        num_workers=args.num_workers,
        collate_fn=collate_campaign_bids,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_campaign_bids,
        pin_memory=device.type == "cuda",
    )

    # Infer input_dim from data
    sample_batch = next(iter(train_loader))
    campaign_dim = sample_batch["campaign_features"].shape[-1]
    context_dim = sample_batch["context_features"].shape[-1]
    input_dim = campaign_dim + context_dim

    model = AutobidNet(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        cross_layers=config.cross_layers,
        dropout=config.dropout,
        min_bid=config.bid_range[0],
        max_bid=config.bid_range[1],
    )

    loss_fn = AutobidLoss(
        constraint_penalty_weight=config.constraint_penalty_weight,
        entropy_weight=config.entropy_weight,
        min_bid=config.bid_range[0],
        max_bid=config.bid_range[1],
    )

    trainer = AutobidTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        grad_clip_norm=config.grad_clip_norm,
        early_stopping_patience=config.early_stopping_patience,
        replay_buffer_size=config.replay_buffer_size,
        replay_batch_size=config.replay_batch_size,
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
        logger.info("Training complete. Best val_loss=%.4f", min(history["val_loss"]))

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
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12356")
        mp.spawn(train_worker, args=(world_size, args, config), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args, config)


if __name__ == "__main__":
    main()
