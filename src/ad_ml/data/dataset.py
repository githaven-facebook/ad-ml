"""PyTorch Dataset classes for user behavior sequences and campaign bid data."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class UserBehaviorDataset(Dataset[Dict[str, Tensor]]):
    """Dataset of user action sequences with temporal features.

    Each sample represents one user's interaction history as a variable-length
    sequence of action embeddings plus user-level categorical attributes.

    Args:
        sequences: List of 2-D arrays of shape (seq_len, feature_dim).
        user_features: 2-D array of shape (n_users, n_user_features).
        labels: Optional 2-D array of shape (n_users, label_dim) for supervised tasks.
        max_length: Truncate sequences longer than this (truncate from left, keep recent).
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        user_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        max_length: int = 512,
    ) -> None:
        if len(sequences) != len(user_features):
            raise ValueError(
                f"sequences ({len(sequences)}) and user_features ({len(user_features)}) must have same length"
            )
        if labels is not None and len(labels) != len(sequences):
            raise ValueError("labels must have same length as sequences")
        self.sequences = sequences
        self.user_features = user_features.astype(np.float32)
        self.labels = labels.astype(np.float32) if labels is not None else None
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        seq = self.sequences[idx].astype(np.float32)
        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]
        sample: Dict[str, Tensor] = {
            "sequence": torch.from_numpy(seq),
            "seq_length": torch.tensor(len(seq), dtype=torch.long),
            "user_features": torch.from_numpy(self.user_features[idx]),
        }
        if self.labels is not None:
            sample["labels"] = torch.from_numpy(self.labels[idx])
        return sample


def collate_user_sequences(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate variable-length sequences into padded tensors.

    Returns a dict with:
        - sequences: (B, T_max, F) float tensor, left-zero-padded
        - seq_lengths: (B,) long tensor with actual sequence lengths
        - user_features: (B, U) float tensor
        - attention_mask: (B, T_max) bool tensor, True where valid
        - labels: (B, L) float tensor (if present in batch)
    """
    seq_lengths = torch.stack([item["seq_length"] for item in batch])
    max_len = int(seq_lengths.max().item())
    feature_dim = batch[0]["sequence"].shape[-1]
    bsz = len(batch)

    padded_seqs = torch.zeros(bsz, max_len, feature_dim, dtype=torch.float32)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq = item["sequence"]
        length = seq.shape[0]
        padded_seqs[i, -length:] = seq
        attention_mask[i, -length:] = True

    result: Dict[str, Tensor] = {
        "sequences": padded_seqs,
        "seq_lengths": seq_lengths,
        "user_features": torch.stack([item["user_features"] for item in batch]),
        "attention_mask": attention_mask,
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


class CampaignBidDataset(Dataset[Dict[str, Tensor]]):
    """Dataset of campaign performance records with bid labels.

    Each sample is one campaign-auction observation with features derived from
    historical performance and contextual signals, plus the optimal bid multiplier
    as supervision signal.

    Args:
        campaign_features: 2-D array of shape (n_samples, n_campaign_features).
        context_features: 2-D array of shape (n_samples, n_context_features).
        bid_labels: 1-D array of shape (n_samples,) with optimal bid multipliers.
        budget_utilizations: Optional 1-D array of budget utilization rates for constraints.
    """

    def __init__(
        self,
        campaign_features: np.ndarray,
        context_features: np.ndarray,
        bid_labels: np.ndarray,
        budget_utilizations: Optional[np.ndarray] = None,
    ) -> None:
        n = len(campaign_features)
        if len(context_features) != n or len(bid_labels) != n:
            raise ValueError("All arrays must have the same number of samples")
        if budget_utilizations is not None and len(budget_utilizations) != n:
            raise ValueError("budget_utilizations must have same length as other arrays")

        self.campaign_features = campaign_features.astype(np.float32)
        self.context_features = context_features.astype(np.float32)
        self.bid_labels = bid_labels.astype(np.float32)
        self.budget_utilizations = (
            budget_utilizations.astype(np.float32) if budget_utilizations is not None else None
        )

    def __len__(self) -> int:
        return len(self.campaign_features)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample: Dict[str, Tensor] = {
            "campaign_features": torch.from_numpy(self.campaign_features[idx]),
            "context_features": torch.from_numpy(self.context_features[idx]),
            "bid_label": torch.tensor(self.bid_labels[idx], dtype=torch.float32),
        }
        if self.budget_utilizations is not None:
            sample["budget_utilization"] = torch.tensor(
                self.budget_utilizations[idx], dtype=torch.float32
            )
        return sample


def collate_campaign_bids(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Standard collate for campaign bid samples (all fixed-size, just stack)."""
    result: Dict[str, Tensor] = {
        "campaign_features": torch.stack([item["campaign_features"] for item in batch]),
        "context_features": torch.stack([item["context_features"] for item in batch]),
        "bid_label": torch.stack([item["bid_label"] for item in batch]),
    }
    if "budget_utilization" in batch[0]:
        result["budget_utilization"] = torch.stack(
            [item["budget_utilization"] for item in batch]
        )
    return result
