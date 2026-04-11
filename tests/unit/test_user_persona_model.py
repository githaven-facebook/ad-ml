"""Unit tests for UserPersonaNet forward pass, loss, and output shapes."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from ad_ml.data.dataset import collate_user_sequences
from ad_ml.models.user_persona.model import PersonaLoss, PersonaOutput, UserPersonaNet
from tests.conftest import (
    EMBEDDING_DIM,
    NUM_SEGMENTS,
    NUM_USERS,
    SEQ_FEATURE_DIM,
    USER_FEATURE_DIM,
)


class TestUserPersonaNetForward:
    def test_output_types(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output: PersonaOutput = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert isinstance(output, PersonaOutput)
        assert isinstance(output.user_embedding, torch.Tensor)
        assert isinstance(output.cluster_probs, torch.Tensor)

    def test_embedding_shape(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        bsz = batch["sequences"].shape[0]
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert output.user_embedding.shape == (bsz, EMBEDDING_DIM), (
            f"Expected ({bsz}, {EMBEDDING_DIM}), got {output.user_embedding.shape}"
        )

    def test_cluster_probs_shape(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        bsz = batch["sequences"].shape[0]
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert output.cluster_probs.shape == (bsz, NUM_SEGMENTS)

    def test_cluster_probs_sum_to_one(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
            gumbel_temperature=1.0,
            hard_gumbel=False,
        )
        probs_sum = output.cluster_probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), (
            f"Cluster probs should sum to 1, got {probs_sum}"
        )

    def test_embedding_l2_normalized(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        norms = torch.norm(output.user_embedding, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Embeddings should be L2 normalized, got norms: {norms}"
        )

    def test_reconstruction_shape(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        bsz = batch["sequences"].shape[0]
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert output.reconstructed.shape == (bsz, SEQ_FEATURE_DIM)

    def test_attention_weights_shape(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        bsz = batch["sequences"].shape[0]
        T = batch["sequences"].shape[1]
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert output.attention_weights.shape == (bsz, T)

    def test_no_nan_in_output(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        assert not torch.isnan(output.user_embedding).any(), "NaN in user_embedding"
        assert not torch.isnan(output.cluster_probs).any(), "NaN in cluster_probs"


class TestPersonaLoss:
    def test_loss_keys(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        loss_fn = PersonaLoss()
        seq_mean = batch["sequences"].mean(dim=1)
        losses = loss_fn(output, seq_mean)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "clustering" in losses
        assert "contrastive" in losses

    def test_loss_is_scalar(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        loss_fn = PersonaLoss()
        seq_mean = batch["sequences"].mean(dim=1)
        losses = loss_fn(output, seq_mean)
        assert losses["total"].shape == torch.Size([])

    def test_loss_positive(self, persona_model: UserPersonaNet, user_behavior_dataset):
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = persona_model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        loss_fn = PersonaLoss()
        seq_mean = batch["sequences"].mean(dim=1)
        losses = loss_fn(output, seq_mean)
        assert losses["total"].item() >= 0.0

    def test_loss_backprop(self, persona_model: UserPersonaNet, user_behavior_dataset):
        """Verify that loss can be back-propagated without errors."""
        # Create a separate model instance to avoid affecting session-scoped fixture
        model = UserPersonaNet(
            seq_feature_dim=SEQ_FEATURE_DIM,
            user_feature_dim=USER_FEATURE_DIM,
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=[64, 32],
            num_segments=NUM_SEGMENTS,
            gru_hidden_size=64,
            gru_num_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        loader = DataLoader(
            user_behavior_dataset,
            batch_size=8,
            collate_fn=collate_user_sequences,
        )
        batch = next(iter(loader))
        output = model(
            sequences=batch["sequences"],
            user_features=batch["user_features"],
            seq_lengths=batch["seq_lengths"],
        )
        loss_fn = PersonaLoss()
        seq_mean = batch["sequences"].mean(dim=1)
        losses = loss_fn(output, seq_mean)
        losses["total"].backward()
        # Check that gradients exist for at least one parameter
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed during backward pass"
