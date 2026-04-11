"""Unit tests for AutobidNet forward pass, bid constraints, and loss computation."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from ad_ml.data.dataset import collate_campaign_bids
from ad_ml.models.autobid.model import AutobidLoss, AutobidNet, AutobidOutput
from tests.conftest import CAMPAIGN_FEATURE_DIM, CONTEXT_FEATURE_DIM, NUM_CAMPAIGNS

MIN_BID = 0.5
MAX_BID = 3.0
INPUT_DIM = CAMPAIGN_FEATURE_DIM + CONTEXT_FEATURE_DIM


class TestAutobidNetForward:
    def test_output_type(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output: AutobidOutput = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        assert isinstance(output, AutobidOutput)

    def test_bid_multiplier_shape(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        bsz = batch["campaign_features"].shape[0]
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        assert output.bid_multiplier.shape == (bsz,)

    def test_bid_range_constraint(self, autobid_model: AutobidNet, campaign_bid_dataset):
        """Bid multipliers must always be in [min_bid, max_bid]."""
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        bids = output.bid_multiplier
        assert (bids >= MIN_BID).all(), f"Some bids below min_bid={MIN_BID}: {bids.min()}"
        assert (bids <= MAX_BID).all(), f"Some bids above max_bid={MAX_BID}: {bids.max()}"

    def test_bid_range_with_budget_utilization(self, autobid_model: AutobidNet, campaign_bid_dataset):
        """Bid range constraint should hold even with budget utilization signals."""
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
            budget_utilization=batch.get("budget_utilization"),
        )
        bids = output.bid_multiplier
        assert (bids >= MIN_BID).all()
        assert (bids <= MAX_BID).all()

    def test_cross_features_shape(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        bsz = batch["campaign_features"].shape[0]
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        assert output.cross_features.shape == (bsz, INPUT_DIM), (
            f"Expected ({bsz}, {INPUT_DIM}), got {output.cross_features.shape}"
        )

    def test_no_nan_in_output(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        assert not torch.isnan(output.bid_multiplier).any(), "NaN in bid_multiplier"
        assert not torch.isnan(output.cross_features).any(), "NaN in cross_features"

    def test_high_budget_utilization_lowers_bids(self, autobid_model: AutobidNet):
        """When budget utilization is 1.0, bids should be <= bids at utilization 0.0."""
        cf = torch.ones(8, CAMPAIGN_FEATURE_DIM)
        ctx = torch.ones(8, CONTEXT_FEATURE_DIM)
        bu_low = torch.zeros(8)   # No budget pressure
        bu_high = torch.ones(8)   # Maximum budget pressure

        with torch.no_grad():
            out_low = autobid_model(cf, ctx, bu_low)
            out_high = autobid_model(cf, ctx, bu_high)

        # With max budget pressure, bids should be <= low-pressure bids
        assert (out_high.bid_multiplier <= out_low.bid_multiplier + 1e-5).all(), (
            "High budget utilization should not increase bids"
        )


class TestAutobidLoss:
    def test_loss_keys(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        loss_fn = AutobidLoss(min_bid=MIN_BID, max_bid=MAX_BID)
        losses = loss_fn(output, batch["bid_label"])
        assert set(losses.keys()) == {"total", "primary", "constraint", "entropy"}

    def test_loss_scalar(self, autobid_model: AutobidNet, campaign_bid_dataset):
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = autobid_model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
        )
        loss_fn = AutobidLoss(min_bid=MIN_BID, max_bid=MAX_BID)
        losses = loss_fn(output, batch["bid_label"])
        assert losses["total"].shape == torch.Size([])

    def test_loss_backprop(self, campaign_bid_dataset):
        """Verify backpropagation through the full loss."""
        model = AutobidNet(
            input_dim=INPUT_DIM,
            hidden_dims=[64, 32],
            cross_layers=2,
            dropout=0.0,
            min_bid=MIN_BID,
            max_bid=MAX_BID,
        )
        loader = DataLoader(
            campaign_bid_dataset,
            batch_size=16,
            collate_fn=collate_campaign_bids,
        )
        batch = next(iter(loader))
        output = model(
            campaign_features=batch["campaign_features"],
            context_features=batch["context_features"],
            budget_utilization=batch.get("budget_utilization"),
        )
        loss_fn = AutobidLoss(min_bid=MIN_BID, max_bid=MAX_BID)
        losses = loss_fn(output, batch["bid_label"], batch.get("budget_utilization"))
        losses["total"].backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
