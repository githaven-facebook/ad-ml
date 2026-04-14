"""Contextual feature extraction: temporal, geo, device, and placement signals."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# US federal holidays (month, day) for is_holiday feature
_US_HOLIDAYS: List[tuple[int, int]] = [
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (11, 11), # Veterans Day
    (12, 25), # Christmas Day
]

# Vocabulary defaults
_DEFAULT_GEO_REGIONS = ["US-CA", "US-NY", "US-TX", "US-FL", "US-WA", "OTHER"]
_DEFAULT_DEVICE_TYPES = ["mobile", "desktop", "tablet", "ctv"]
_DEFAULT_CONNECTION_TYPES = ["wifi", "4g", "5g", "3g", "unknown"]
_DEFAULT_APP_CATEGORIES = [
    "news", "sports", "entertainment", "gaming", "finance",
    "shopping", "travel", "food", "health", "social", "other",
]


class ContextFeatureExtractor:
    """Extract contextual features for ad auction requests.

    Features:
    - Temporal: sinusoidal hour-of-day encoding, day-of-week one-hot,
                is_weekend flag, is_holiday flag
    - Geo: one-hot region embedding
    - Device: one-hot device type
    - Connection: one-hot connection type
    - Placement: one-hot app/site category
    """

    def __init__(
        self,
        geo_vocab: Optional[List[str]] = None,
        device_vocab: Optional[List[str]] = None,
        connection_vocab: Optional[List[str]] = None,
        app_category_vocab: Optional[List[str]] = None,
    ) -> None:
        self.geo_vocab = geo_vocab or _DEFAULT_GEO_REGIONS
        self.device_vocab = device_vocab or _DEFAULT_DEVICE_TYPES
        self.connection_vocab = connection_vocab or _DEFAULT_CONNECTION_TYPES
        self.app_category_vocab = app_category_vocab or _DEFAULT_APP_CATEGORIES

        self._geo_idx = {v: i for i, v in enumerate(self.geo_vocab)}
        self._device_idx = {v: i for i, v in enumerate(self.device_vocab)}
        self._conn_idx = {v: i for i, v in enumerate(self.connection_vocab)}
        self._app_idx = {v: i for i, v in enumerate(self.app_category_vocab)}

    @property
    def feature_dim(self) -> int:
        """Total dimension of the context feature vector."""
        # temporal: 4 (sin_hour, cos_hour, is_weekend, is_holiday) + 7 (dow one-hot)
        temporal_dim = 4 + 7
        return (
            temporal_dim
            + len(self.geo_vocab)
            + len(self.device_vocab)
            + len(self.connection_vocab)
            + len(self.app_category_vocab)
        )

    def extract(self, requests: pd.DataFrame) -> "np.ndarray[Any, np.dtype[np.float32]]":
        """Extract context features from a batch of auction requests.

        Args:
            requests: DataFrame with columns: timestamp (datetime),
                      geo_region (str), device_type (str),
                      connection_type (str), app_category (str).

        Returns:
            Float32 array of shape (n_requests, feature_dim).
        """
        n = len(requests)
        features = np.zeros((n, self.feature_dim), dtype=np.float32)
        col_offset = 0

        # --- Temporal ---
        ts = pd.to_datetime(requests["timestamp"])
        hours = ts.dt.hour.values.astype(np.float32)
        features[:, col_offset + 0] = np.sin(2 * math.pi * hours / 24.0)
        features[:, col_offset + 1] = np.cos(2 * math.pi * hours / 24.0)
        features[:, col_offset + 2] = (ts.dt.dayofweek >= 5).values.astype(np.float32)
        features[:, col_offset + 3] = self._holiday_flags(ts)
        col_offset += 4

        # Day-of-week one-hot (0=Monday)
        dow = ts.dt.dayofweek.values
        for d in range(7):
            features[:, col_offset + d] = (dow == d).astype(np.float32)
        col_offset += 7

        # --- Geo ---
        if "geo_region" in requests.columns:
            col_offset = self._one_hot_col(
                requests["geo_region"].values, self._geo_idx, features, col_offset
            )
        else:
            col_offset += len(self.geo_vocab)

        # --- Device ---
        if "device_type" in requests.columns:
            col_offset = self._one_hot_col(
                requests["device_type"].values, self._device_idx, features, col_offset
            )
        else:
            col_offset += len(self.device_vocab)

        # --- Connection ---
        if "connection_type" in requests.columns:
            col_offset = self._one_hot_col(
                requests["connection_type"].values, self._conn_idx, features, col_offset
            )
        else:
            col_offset += len(self.connection_vocab)

        # --- App category ---
        if "app_category" in requests.columns:
            col_offset = self._one_hot_col(
                requests["app_category"].values, self._app_idx, features, col_offset
            )
        else:
            col_offset += len(self.app_category_vocab)

        return features

    def extract_single(self, record: Dict[str, object]) -> "np.ndarray[Any, np.dtype[np.float32]]":
        """Extract context features for a single auction request dict.

        Returns:
            Float32 array of shape (feature_dim,).
        """
        df = pd.DataFrame([record])
        result: "np.ndarray[Any, np.dtype[np.float32]]" = self.extract(df)[0]
        return result

    def _holiday_flags(self, timestamps: pd.Series) -> "np.ndarray[Any, np.dtype[np.float32]]":
        """Return 1.0 for dates that fall on a US federal holiday."""
        flags = np.zeros(len(timestamps), dtype=np.float32)
        months = timestamps.dt.month.values
        days = timestamps.dt.day.values
        for month, day in _US_HOLIDAYS:
            flags |= ((months == month) & (days == day)).astype(np.float32)
        return flags

    @staticmethod
    def _one_hot_col(
        values: np.ndarray,
        vocab_idx: Dict[str, int],
        out: np.ndarray,
        offset: int,
    ) -> int:
        """Fill one-hot columns in-place starting at offset. Returns next offset."""
        vocab_size = len(vocab_idx)
        for i, val in enumerate(values):
            idx = vocab_idx.get(str(val), None)
            if idx is not None:
                out[i, offset + idx] = 1.0
        return offset + vocab_size


def build_time_embedding(timestamps: Sequence[pd.Timestamp], dim: int = 16) -> "np.ndarray[Any, np.dtype[np.float32]]":
    """Build sinusoidal time embeddings from timestamps.

    Each timestamp is encoded across multiple frequencies (hour, day, week, month).

    Args:
        timestamps: Sequence of pandas Timestamps.
        dim: Embedding dimension (must be even).

    Returns:
        Float32 array of shape (len(timestamps), dim).
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    n = len(timestamps)
    emb = np.zeros((n, dim), dtype=np.float32)
    periods = [24.0, 168.0, 720.0, 8760.0]  # hours: day, week, month, year
    hours = np.array(
        [(ts - pd.Timestamp("2020-01-01")).total_seconds() / 3600.0 for ts in timestamps],
        dtype=np.float64,
    )
    pairs_per_period = max(1, dim // (2 * len(periods)))
    col = 0
    for period in periods:
        for k in range(1, pairs_per_period + 1):
            if col + 1 >= dim:
                break
            freq = 2 * math.pi * k / period
            emb[:, col] = np.sin(freq * hours).astype(np.float32)
            emb[:, col + 1] = np.cos(freq * hours).astype(np.float32)
            col += 2
    return emb
