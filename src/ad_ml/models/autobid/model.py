"""AutobidNet: Deep & Cross Network V2 for bid multiplier prediction."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AutobidOutput(NamedTuple):
    """Outputs from a forward pass of AutobidNet."""

    bid_multiplier: Tensor    # (B,) scaled to [min_bid, max_bid]
    bid_raw: Tensor           # (B,) unbounded pre-constraint output
    cross_features: Tensor    # (B, input_dim) learned cross interactions
    deep_features: Tensor     # (B, deep_output_dim) deep representation


class CrossLayer(nn.Module):
    """Single DCN-V2 cross layer: x_{l+1} = x_0 * W_l * x_l + b_l + x_l."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x0: Tensor, xl: Tensor) -> Tensor:
        """
        Args:
            x0: (B, D) — original input, fixed across all cross layers
            xl: (B, D) — input from previous cross layer

        Returns:
            x_{l+1}: (B, D)
        """
        return x0 * (xl @ self.weight + self.bias) + xl


class CrossNetwork(nn.Module):
    """Stack of DCN-V2 cross layers."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D)

        Returns:
            x_cross: (B, D)
        """
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class BudgetPacingConstraint(nn.Module):
    """Soft constraint layer that adjusts bid multipliers based on budget utilization.

    When budget is nearly exhausted (utilization -> 1), multiplier is scaled down
    to reduce spend. This is differentiable so the constraint is learned.
    """

    def __init__(self, min_bid: float = 0.5, max_bid: float = 3.0) -> None:
        super().__init__()
        self.min_bid = min_bid
        self.max_bid = max_bid
        # Learnable pacing sensitivity
        self.pacing_slope = nn.Parameter(torch.tensor(2.0))

    def forward(
        self, raw_bid: Tensor, budget_utilization: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            raw_bid: (B,) unbounded bid logits
            budget_utilization: (B,) in [0, 1], fraction of budget spent

        Returns:
            bid_multiplier: (B,) scaled to [min_bid, max_bid]
        """
        # Scale raw bid to [min_bid, max_bid] using sigmoid
        bid = self.min_bid + (self.max_bid - self.min_bid) * torch.sigmoid(raw_bid)

        if budget_utilization is not None:
            # Pacing multiplier: approaches 0 as utilization -> 1
            pacing = torch.exp(-F.softplus(self.pacing_slope) * budget_utilization)
            # Blend: keep full bid when util=0, scale down when util->1
            min_bid_tensor = torch.tensor(
                self.min_bid, device=bid.device, dtype=bid.dtype
            )
            bid = torch.maximum(bid * pacing, min_bid_tensor)

        return bid.clamp(self.min_bid, self.max_bid)


class AutobidNet(nn.Module):
    """Deep & Cross Network V2 for autobid multiplier prediction.

    Architecture:
        1. Input normalization layer.
        2. Cross network (explicit high-order interactions).
        3. Deep network (implicit feature learning, MLP).
        4. Stacked output: cross_out || deep_out -> final MLP.
        5. Budget pacing constraint layer -> bid_multiplier in [min_bid, max_bid].
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: int = 3,
        dropout: float = 0.1,
        min_bid: float = 0.5,
        max_bid: float = 3.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        hidden_dims = hidden_dims or [512, 256, 128]
        self.min_bid = min_bid
        self.max_bid = max_bid

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Cross network
        self.cross_network = CrossNetwork(input_dim, num_layers=cross_layers)

        # Deep network
        deep_layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            deep_layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        self.deep_network = nn.Sequential(*deep_layers)
        self.deep_output_dim = prev_dim

        # Stacked output head: cross || deep -> bid scalar
        stacked_dim = input_dim + self.deep_output_dim
        self.output_head = nn.Sequential(
            nn.Linear(stacked_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Budget pacing constraint
        self.pacing_constraint = BudgetPacingConstraint(min_bid=min_bid, max_bid=max_bid)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        campaign_features: Tensor,
        context_features: Tensor,
        budget_utilization: Optional[Tensor] = None,
    ) -> AutobidOutput:
        """Forward pass.

        Args:
            campaign_features: (B, campaign_feature_dim)
            context_features: (B, context_feature_dim)
            budget_utilization: (B,) optional, fraction of budget spent [0, 1]

        Returns:
            AutobidOutput named tuple.
        """
        # Concatenate all inputs
        x = torch.cat([campaign_features, context_features], dim=-1)  # (B, input_dim)
        x = self.input_norm(x)

        # Cross path
        cross_out = self.cross_network(x)  # (B, input_dim)

        # Deep path
        deep_out = self.deep_network(x)  # (B, deep_output_dim)

        # Stack and predict
        stacked = torch.cat([cross_out, deep_out], dim=-1)  # (B, input_dim + deep_dim)
        bid_raw = self.output_head(stacked).squeeze(-1)  # (B,)

        # Apply pacing constraint
        bid_multiplier = self.pacing_constraint(bid_raw, budget_utilization)

        return AutobidOutput(
            bid_multiplier=bid_multiplier,
            bid_raw=bid_raw,
            cross_features=cross_out,
            deep_features=deep_out,
        )


class AutobidLoss(nn.Module):
    """Combined loss for autobidding training.

    Components:
    - Primary: Huber loss on bid multiplier (robust to outliers vs pure MSE).
    - Constraint penalty: penalize predicted bids that would cause budget overspend.
    - Entropy regularization: encourage exploration over the bid range.
    """

    def __init__(
        self,
        constraint_penalty_weight: float = 0.5,
        entropy_weight: float = 0.01,
        min_bid: float = 0.5,
        max_bid: float = 3.0,
        huber_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.constraint_penalty_weight = constraint_penalty_weight
        self.entropy_weight = entropy_weight
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        output: AutobidOutput,
        bid_labels: Tensor,
        budget_utilization: Optional[Tensor] = None,
        target_utilization: float = 0.9,
    ) -> Dict[str, Tensor]:
        """Compute the combined autobidding loss.

        Args:
            output: AutobidOutput from AutobidNet.forward().
            bid_labels: (B,) ground-truth optimal bid multipliers.
            budget_utilization: (B,) current budget utilization rates.
            target_utilization: Desired max budget utilization rate.

        Returns:
            Dict with keys 'total', 'primary', 'constraint', 'entropy'.
        """
        # Primary: Huber regression loss
        primary_loss = self.huber(output.bid_multiplier, bid_labels)

        # Constraint penalty: penalize when predicted bids would overspend budget
        constraint_loss = torch.tensor(0.0, device=output.bid_multiplier.device)
        if budget_utilization is not None:
            overspend_mask = budget_utilization > target_utilization
            if overspend_mask.any():
                overbudget_bids = output.bid_multiplier[overspend_mask]
                # Penalize bids above 1.0 (neutral) when over budget
                penalty = F.relu(overbudget_bids - 1.0).pow(2).mean()
                constraint_loss = penalty

        # Entropy regularization: encourage spread across bid range
        bid_normalized = (output.bid_multiplier - self.min_bid) / (
            self.max_bid - self.min_bid
        )
        entropy_loss = -torch.log(bid_normalized.clamp(1e-6, 1 - 1e-6)).mean()

        total = (
            primary_loss
            + self.constraint_penalty_weight * constraint_loss
            + self.entropy_weight * entropy_loss
        )

        return {
            "total": total,
            "primary": primary_loss,
            "constraint": constraint_loss,
            "entropy": entropy_loss,
        }
