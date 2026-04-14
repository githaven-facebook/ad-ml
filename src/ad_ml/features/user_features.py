"""User behavior feature extraction from event streams."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Action type constants
ACTION_IMPRESSION = "impression"
ACTION_CLICK = "click"
ACTION_CONVERSION = "conversion"

# Time window definitions in hours
TIME_WINDOWS = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}


class UserFeatureExtractor:
    """Extract behavioral features for each user from raw event logs.

    Features include:
    - Action counts per time window (impressions, clicks, conversions)
    - Recency scores (exponential decay from last event)
    - Frequency patterns (inter-event time statistics)
    - Category affinity vector (normalized click counts per content category)
    - Time-of-day activity distribution (24-bin histogram)
    - Device preference distribution
    """

    def __init__(
        self,
        category_vocab: Optional[List[str]] = None,
        device_vocab: Optional[List[str]] = None,
        reference_time: Optional[pd.Timestamp] = None,
    ) -> None:
        self.category_vocab = category_vocab or []
        self.device_vocab = device_vocab or ["mobile", "desktop", "tablet"]
        self.reference_time = reference_time

    def extract(self, events: pd.DataFrame, user_id_col: str = "user_id") -> pd.DataFrame:
        """Extract features for all users in an event DataFrame.

        Args:
            events: DataFrame with columns user_id, event_type, category, device,
                    event_time (datetime), and optionally revenue.
            user_id_col: Name of the user identifier column.

        Returns:
            DataFrame indexed by user_id with all feature columns.
        """
        events = events.copy()
        events["event_time"] = pd.to_datetime(events["event_time"])
        ref_time = self.reference_time or events["event_time"].max()

        feature_frames: List[pd.DataFrame] = []

        # Action counts per time window
        feature_frames.append(self._action_counts(events, user_id_col, ref_time))

        # Recency scores
        feature_frames.append(self._recency_scores(events, user_id_col, ref_time))

        # Frequency patterns
        feature_frames.append(self._frequency_patterns(events, user_id_col))

        # Category affinity
        if self.category_vocab and "category" in events.columns:
            feature_frames.append(
                self._category_affinity(events, user_id_col)
            )

        # Time-of-day distribution
        feature_frames.append(self._time_of_day_distribution(events, user_id_col))

        # Device preferences
        if "device" in events.columns:
            feature_frames.append(self._device_preferences(events, user_id_col))

        result = feature_frames[0]
        for df in feature_frames[1:]:
            result = result.join(df, how="outer")
        return result.fillna(0.0)

    def _action_counts(
        self,
        events: pd.DataFrame,
        user_id_col: str,
        ref_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Compute impression/click/conversion counts per time window."""
        rows: Dict[str, Dict[str, float]] = {}
        for window_name, hours in TIME_WINDOWS.items():
            cutoff = ref_time - pd.Timedelta(hours=hours)
            window_events = events[events["event_time"] >= cutoff]
            counts = (
                window_events.groupby([user_id_col, "event_type"])
                .size()
                .unstack(fill_value=0)
            )
            for action in [ACTION_IMPRESSION, ACTION_CLICK, ACTION_CONVERSION]:
                col_name = f"count_{action}_{window_name}"
                for uid, row in counts.iterrows():
                    rows.setdefault(str(uid), {})[col_name] = float(
                        row.get(action, 0)
                    )
        return pd.DataFrame.from_dict(rows, orient="index")

    def _recency_scores(
        self,
        events: pd.DataFrame,
        user_id_col: str,
        ref_time: pd.Timestamp,
        decay_rate: float = 0.1,
    ) -> pd.DataFrame:
        """Compute exponential recency score for each action type."""
        scores: Dict[str, Dict[str, float]] = {}
        for action in [ACTION_IMPRESSION, ACTION_CLICK, ACTION_CONVERSION]:
            action_events = events[events["event_type"] == action]
            if action_events.empty:
                continue
            last_time = action_events.groupby(user_id_col)["event_time"].max()
            hours_since = (ref_time - last_time).dt.total_seconds() / 3600.0
            recency = np.exp(-decay_rate * hours_since)
            for uid, score in recency.items():
                scores.setdefault(str(uid), {})[f"recency_{action}"] = float(score)
        return pd.DataFrame.from_dict(scores, orient="index")

    def _frequency_patterns(
        self, events: pd.DataFrame, user_id_col: str
    ) -> pd.DataFrame:
        """Compute mean and std of inter-event times in hours."""
        results: Dict[str, Dict[str, float]] = {}
        for uid, user_events in events.groupby(user_id_col):
            times = user_events["event_time"].sort_values()
            diffs = times.diff().dt.total_seconds().dropna() / 3600.0
            results[str(uid)] = {
                "inter_event_mean_h": float(diffs.mean()) if len(diffs) > 0 else 0.0,
                "inter_event_std_h": float(diffs.std()) if len(diffs) > 1 else 0.0,
                "total_events": float(len(times)),
            }
        return pd.DataFrame.from_dict(results, orient="index")

    def _category_affinity(
        self, events: pd.DataFrame, user_id_col: str
    ) -> pd.DataFrame:
        """Compute normalized click-through rate per content category."""
        clicks = events[events["event_type"] == ACTION_CLICK]
        if clicks.empty or not self.category_vocab:
            return pd.DataFrame()
        counts = (
            clicks.groupby([user_id_col, "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=self.category_vocab, fill_value=0)
        )
        row_sums = counts.sum(axis=1).replace(0, 1)
        normalized = counts.div(row_sums, axis=0)
        normalized.columns = [f"cat_affinity_{c}" for c in normalized.columns]
        normalized.index = normalized.index.astype(str)
        return normalized

    def _time_of_day_distribution(
        self, events: pd.DataFrame, user_id_col: str
    ) -> pd.DataFrame:
        """Compute 24-bin activity histogram normalized per user."""
        events = events.copy()
        events["hour"] = events["event_time"].dt.hour
        pivot = (
            events.groupby([user_id_col, "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(range(24)), fill_value=0)
        )
        row_sums = pivot.sum(axis=1).replace(0, 1)
        normalized = pivot.div(row_sums, axis=0)
        normalized.columns = [f"tod_hour_{h:02d}" for h in range(24)]
        normalized.index = normalized.index.astype(str)
        return normalized

    def _device_preferences(
        self, events: pd.DataFrame, user_id_col: str
    ) -> pd.DataFrame:
        """Compute normalized device type distribution per user."""
        pivot = (
            events.groupby([user_id_col, "device"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=self.device_vocab, fill_value=0)
        )
        row_sums = pivot.sum(axis=1).replace(0, 1)
        normalized = pivot.div(row_sums, axis=0)
        normalized.columns = [f"device_{d}" for d in self.device_vocab]
        normalized.index = normalized.index.astype(str)
        return normalized


def build_user_sequence(
    user_events: pd.DataFrame,
    feature_cols: List[str],
    max_length: int = 512,
) -> "np.ndarray[Any, np.dtype[np.float32]]":
    """Build a temporal feature sequence array for a single user.

    Args:
        user_events: Events for one user sorted by event_time.
        feature_cols: Feature columns to include in each timestep vector.
        max_length: Truncate to this many events (keep most recent).

    Returns:
        Array of shape (min(n_events, max_length), len(feature_cols)).
    """
    user_events = user_events.sort_values("event_time")
    if len(user_events) > max_length:
        user_events = user_events.iloc[-max_length:]
    result: np.ndarray[Any, np.dtype[np.float32]] = user_events[feature_cols].values.astype(np.float32)
    return result
