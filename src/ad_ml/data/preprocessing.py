"""Feature preprocessing: normalization, imputation, encoding, sequence padding, and splitting."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def normalize_features(
    df: pd.DataFrame,
    columns: List[str],
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Z-score normalize specified numeric columns.

    Args:
        df: Input DataFrame.
        columns: Columns to normalize.
        scaler: Pre-fitted scaler; if None a new one is created.
        fit: Whether to fit the scaler (False for val/test).

    Returns:
        Transformed DataFrame and the fitted scaler.
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        df[columns] = scaler.fit_transform(df[columns].astype(np.float32))
    else:
        df[columns] = scaler.transform(df[columns].astype(np.float32))
    return df, scaler


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_fill: str = "__MISSING__",
) -> pd.DataFrame:
    """Impute missing values.

    Args:
        df: Input DataFrame.
        numeric_strategy: Strategy for numeric columns ("median" or "mean").
        categorical_fill: Fill value for object/category columns.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_strategy == "median":
        fill_values = {col: df[col].median() for col in numeric_cols}
    elif numeric_strategy == "mean":
        fill_values = {col: df[col].mean() for col in numeric_cols}
    else:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}")

    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    df[cat_cols] = df[cat_cols].fillna(categorical_fill)
    return df


def encode_categoricals(
    df: pd.DataFrame,
    columns: List[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Ordinal-encode categorical columns.

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.
        encoders: Pre-fitted encoders; if None new ones are created.
        fit: Whether to fit the encoders.

    Returns:
        Encoded DataFrame and the dict of fitted encoders.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in columns:
        if col not in encoders:
            encoders[col] = LabelEncoder()
        if fit:
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        else:
            # Handle unseen labels by mapping to 0
            known = set(encoders[col].classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else encoders[col].classes_[0])
            df[col] = encoders[col].transform(df[col])
    return df, encoders


def pad_sequences(
    sequences: List[List[float]],
    max_length: int,
    pad_value: float = 0.0,
    truncate_side: str = "left",
) -> np.ndarray:
    """Pad or truncate a list of variable-length sequences to a fixed length.

    Args:
        sequences: List of 1-D sequences.
        max_length: Target length.
        pad_value: Value used for padding.
        truncate_side: "left" truncates oldest events (keep recent), "right" truncates newest.

    Returns:
        2-D array of shape (len(sequences), max_length).
    """
    out = np.full((len(sequences), max_length), pad_value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) > max_length:
            if truncate_side == "left":
                seq = seq[-max_length:]
            else:
                seq = seq[:max_length]
        out[i, -len(seq):] = seq
    return out


class FeaturePreprocessor:
    """Stateful preprocessor that encapsulates fitting and transforming all feature types."""

    def __init__(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        numeric_strategy: str = "median",
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_strategy = numeric_strategy
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed copy."""
        df = impute_missing(df, numeric_strategy=self.numeric_strategy)
        if self.numeric_cols:
            df, self.scaler = normalize_features(df, self.numeric_cols, fit=True)
        if self.categorical_cols:
            df, self.encoders = encode_categoricals(df, self.categorical_cols, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform df using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor must be fit before calling transform()")
        df = impute_missing(df, numeric_strategy=self.numeric_strategy)
        if self.numeric_cols and self.scaler is not None:
            df, _ = normalize_features(df, self.numeric_cols, scaler=self.scaler, fit=False)
        if self.categorical_cols:
            df, _ = encode_categoricals(df, self.categorical_cols, encoders=self.encoders, fit=False)
        return df


class TemporalSplitter:
    """Split DataFrame into train/val/test preserving temporal ordering.

    Splits are determined by timestamp column percentiles to avoid leakage.
    """

    def __init__(
        self,
        timestamp_col: str = "event_time",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> None:
        if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
            raise ValueError("Fractions must be positive and sum to less than 1")
        self.timestamp_col = timestamp_col
        self.train_frac = train_frac
        self.val_frac = val_frac

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, val, test) DataFrames sorted by timestamp."""
        df_sorted = df.sort_values(self.timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))
        train = df_sorted.iloc[:train_end].copy()
        val = df_sorted.iloc[train_end:val_end].copy()
        test = df_sorted.iloc[val_end:].copy()
        logger.info(
            "Temporal split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
        )
        return train, val, test
