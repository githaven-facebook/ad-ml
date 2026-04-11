"""Shared pytest fixtures for ad-ml tests."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from ad_ml.data.dataset import CampaignBidDataset, UserBehaviorDataset
from ad_ml.models.autobid.model import AutobidNet
from ad_ml.models.user_persona.model import UserPersonaNet

# -----------------------------------------------------------------------
# Dimension constants for all tests
# -----------------------------------------------------------------------
SEQ_FEATURE_DIM = 16
USER_FEATURE_DIM = 8
CAMPAIGN_FEATURE_DIM = 32
CONTEXT_FEATURE_DIM = 16
NUM_USERS = 32
NUM_CAMPAIGNS = 64
SEQ_LEN = 10
NUM_SEGMENTS = 4
EMBEDDING_DIM = 32


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_sequences(rng: np.random.Generator) -> List[np.ndarray]:
    """Variable-length user action sequences."""
    lengths = rng.integers(3, SEQ_LEN + 1, size=NUM_USERS)
    return [rng.random((int(l), SEQ_FEATURE_DIM), dtype=np.float32) for l in lengths]


@pytest.fixture(scope="session")
def sample_user_features(rng: np.random.Generator) -> np.ndarray:
    return rng.random((NUM_USERS, USER_FEATURE_DIM), dtype=np.float32)


@pytest.fixture(scope="session")
def user_behavior_dataset(
    sample_sequences: List[np.ndarray],
    sample_user_features: np.ndarray,
) -> UserBehaviorDataset:
    return UserBehaviorDataset(
        sample_sequences,
        sample_user_features,
        max_length=SEQ_LEN,
    )


@pytest.fixture(scope="session")
def sample_campaign_features(rng: np.random.Generator) -> np.ndarray:
    return rng.random((NUM_CAMPAIGNS, CAMPAIGN_FEATURE_DIM), dtype=np.float32)


@pytest.fixture(scope="session")
def sample_context_features(rng: np.random.Generator) -> np.ndarray:
    return rng.random((NUM_CAMPAIGNS, CONTEXT_FEATURE_DIM), dtype=np.float32)


@pytest.fixture(scope="session")
def sample_bid_labels(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.5, 3.0, size=NUM_CAMPAIGNS).astype(np.float32)


@pytest.fixture(scope="session")
def sample_budget_utils(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=NUM_CAMPAIGNS).astype(np.float32)


@pytest.fixture(scope="session")
def campaign_bid_dataset(
    sample_campaign_features: np.ndarray,
    sample_context_features: np.ndarray,
    sample_bid_labels: np.ndarray,
    sample_budget_utils: np.ndarray,
) -> CampaignBidDataset:
    return CampaignBidDataset(
        sample_campaign_features,
        sample_context_features,
        sample_bid_labels,
        sample_budget_utils,
    )


@pytest.fixture(scope="session")
def persona_model() -> UserPersonaNet:
    return UserPersonaNet(
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


@pytest.fixture(scope="session")
def autobid_model() -> AutobidNet:
    input_dim = CAMPAIGN_FEATURE_DIM + CONTEXT_FEATURE_DIM
    return AutobidNet(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        cross_layers=2,
        dropout=0.0,
        min_bid=0.5,
        max_bid=3.0,
    )


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")
