"""End-to-end integration tests for training pipelines with synthetic data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ad_ml.data.dataset import (
    CampaignBidDataset,
    UserBehaviorDataset,
    collate_campaign_bids,
    collate_user_sequences,
)
from ad_ml.models.autobid.model import AutobidLoss, AutobidNet
from ad_ml.models.autobid.trainer import AutobidTrainer
from ad_ml.models.user_persona.model import PersonaLoss, UserPersonaNet
from ad_ml.models.user_persona.trainer import PersonaTrainer

# Small dimensions for fast integration tests
_SEQ_DIM = 12
_USER_DIM = 8
_CAMP_DIM = 16
_CTX_DIM = 12
_N_TRAIN = 64
_N_VAL = 16
_N_SEGMENTS = 4
_EMB_DIM = 16


def _make_persona_datasets() -> tuple[UserBehaviorDataset, UserBehaviorDataset]:
    rng = np.random.default_rng(0)
    train_seqs: List[np.ndarray] = [
        rng.random((rng.integers(3, 12), _SEQ_DIM), dtype=np.float32)
        for _ in range(_N_TRAIN)
    ]
    val_seqs: List[np.ndarray] = [
        rng.random((rng.integers(3, 12), _SEQ_DIM), dtype=np.float32)
        for _ in range(_N_VAL)
    ]
    train_ufeats = rng.random((_N_TRAIN, _USER_DIM), dtype=np.float32)
    val_ufeats = rng.random((_N_VAL, _USER_DIM), dtype=np.float32)
    train_ds = UserBehaviorDataset(train_seqs, train_ufeats, max_length=16)
    val_ds = UserBehaviorDataset(val_seqs, val_ufeats, max_length=16)
    return train_ds, val_ds


def _make_autobid_datasets() -> tuple[CampaignBidDataset, CampaignBidDataset]:
    rng = np.random.default_rng(1)
    train_cf = rng.random((_N_TRAIN, _CAMP_DIM), dtype=np.float32)
    train_ctx = rng.random((_N_TRAIN, _CTX_DIM), dtype=np.float32)
    train_labels = rng.uniform(0.5, 3.0, size=_N_TRAIN).astype(np.float32)
    train_budget = rng.uniform(0.0, 1.0, size=_N_TRAIN).astype(np.float32)

    val_cf = rng.random((_N_VAL, _CAMP_DIM), dtype=np.float32)
    val_ctx = rng.random((_N_VAL, _CTX_DIM), dtype=np.float32)
    val_labels = rng.uniform(0.5, 3.0, size=_N_VAL).astype(np.float32)
    val_budget = rng.uniform(0.0, 1.0, size=_N_VAL).astype(np.float32)

    train_ds = CampaignBidDataset(train_cf, train_ctx, train_labels, train_budget)
    val_ds = CampaignBidDataset(val_cf, val_ctx, val_labels, val_budget)
    return train_ds, val_ds


class TestPersonaTrainingPipeline:
    def test_two_epoch_training_runs(self, tmp_path: Path):
        """Full training pipeline completes without error for 2 epochs."""
        train_ds, val_ds = _make_persona_datasets()

        train_loader = DataLoader(
            train_ds,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_user_sequences,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_user_sequences,
        )

        model = UserPersonaNet(
            seq_feature_dim=_SEQ_DIM,
            user_feature_dim=_USER_DIM,
            embedding_dim=_EMB_DIM,
            hidden_dims=[32, 16],
            num_segments=_N_SEGMENTS,
            gru_hidden_size=32,
            gru_num_layers=1,
            attention_heads=2,
            dropout=0.0,
        )
        loss_fn = PersonaLoss(reconstruction_weight=1.0, clustering_weight=0.1, contrastive_weight=0.05)

        trainer = PersonaTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            epochs=2,
            warmup_steps=5,
            checkpoint_dir=tmp_path / "checkpoints",
            device=torch.device("cpu"),
            mlflow_experiment="test-persona",
            use_amp=False,
        )

        history = trainer.train(run_name="integration-test")
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(isinstance(v, float) for v in history["train_loss"])
        assert all(not np.isnan(v) for v in history["val_loss"])

    def test_checkpoint_is_saved(self, tmp_path: Path):
        """Best checkpoint is written to disk during training."""
        train_ds, val_ds = _make_persona_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True, collate_fn=collate_user_sequences, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_user_sequences)

        model = UserPersonaNet(
            seq_feature_dim=_SEQ_DIM,
            user_feature_dim=_USER_DIM,
            embedding_dim=_EMB_DIM,
            hidden_dims=[32, 16],
            num_segments=_N_SEGMENTS,
            gru_hidden_size=32,
            gru_num_layers=1,
            attention_heads=2,
            dropout=0.0,
        )
        loss_fn = PersonaLoss()
        ckpt_dir = tmp_path / "checkpoints"
        trainer = PersonaTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            warmup_steps=5,
            checkpoint_dir=ckpt_dir,
            device=torch.device("cpu"),
            mlflow_experiment="test-persona",
            use_amp=False,
        )
        trainer.train()
        assert (ckpt_dir / "best.pt").exists()

    def test_checkpoint_load_and_resume(self, tmp_path: Path):
        """Model state is correctly restored from checkpoint."""
        train_ds, val_ds = _make_persona_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True, collate_fn=collate_user_sequences, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_user_sequences)

        def _build_trainer(ckpt_dir: Path) -> PersonaTrainer:
            model = UserPersonaNet(
                seq_feature_dim=_SEQ_DIM,
                user_feature_dim=_USER_DIM,
                embedding_dim=_EMB_DIM,
                hidden_dims=[32, 16],
                num_segments=_N_SEGMENTS,
                gru_hidden_size=32,
                gru_num_layers=1,
                attention_heads=2,
                dropout=0.0,
            )
            return PersonaTrainer(
                model=model,
                loss_fn=PersonaLoss(),
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=1,
                warmup_steps=5,
                checkpoint_dir=ckpt_dir,
                device=torch.device("cpu"),
                mlflow_experiment="test-persona",
                use_amp=False,
            )

        ckpt_dir = tmp_path / "checkpoints"
        trainer1 = _build_trainer(ckpt_dir)
        trainer1.train()

        # Resume from checkpoint
        trainer2 = _build_trainer(ckpt_dir)
        epoch = trainer2.load_checkpoint(ckpt_dir / "best.pt")
        assert epoch == 1


class TestAutobidTrainingPipeline:
    def test_two_epoch_training_runs(self, tmp_path: Path):
        """Full autobid training completes without error."""
        train_ds, val_ds = _make_autobid_datasets()
        input_dim = _CAMP_DIM + _CTX_DIM

        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True, collate_fn=collate_campaign_bids, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_campaign_bids)

        model = AutobidNet(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            cross_layers=2,
            dropout=0.0,
            min_bid=0.5,
            max_bid=3.0,
        )
        loss_fn = AutobidLoss(min_bid=0.5, max_bid=3.0)

        trainer = AutobidTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            epochs=2,
            warmup_steps=5,
            replay_buffer_size=256,
            replay_batch_size=16,
            checkpoint_dir=tmp_path / "checkpoints",
            device=torch.device("cpu"),
            mlflow_experiment="test-autobid",
            use_amp=False,
        )

        history = trainer.train(run_name="integration-test")
        assert len(history["train_loss"]) == 2
        assert all(not np.isnan(v) for v in history["val_loss"])

    def test_bids_in_range_after_training(self, tmp_path: Path):
        """After training, all predicted bids remain within configured range."""
        train_ds, val_ds = _make_autobid_datasets()
        input_dim = _CAMP_DIM + _CTX_DIM

        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True, collate_fn=collate_campaign_bids, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_campaign_bids)

        model = AutobidNet(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            cross_layers=2,
            dropout=0.0,
            min_bid=0.5,
            max_bid=3.0,
        )
        loss_fn = AutobidLoss(min_bid=0.5, max_bid=3.0)
        trainer = AutobidTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            warmup_steps=5,
            checkpoint_dir=tmp_path / "checkpoints",
            device=torch.device("cpu"),
            mlflow_experiment="test-autobid",
            use_amp=False,
        )
        trainer.train()

        # Check predictions on val set
        model.eval()
        all_bids: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    campaign_features=batch["campaign_features"],
                    context_features=batch["context_features"],
                )
                all_bids.extend(output.bid_multiplier.tolist())

        bids = np.array(all_bids)
        assert (bids >= 0.5).all(), f"Bids below min: {bids.min()}"
        assert (bids <= 3.0).all(), f"Bids above max: {bids.max()}"
