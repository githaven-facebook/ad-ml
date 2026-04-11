"""Unit tests for data preprocessing and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ad_ml.data.preprocessing import (
    FeaturePreprocessor,
    TemporalSplitter,
    encode_categoricals,
    impute_missing,
    normalize_features,
    pad_sequences,
)


class TestNormalizeFeatures:
    def test_basic_normalization(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        result, scaler = normalize_features(df, ["a", "b"], fit=True)
        assert abs(result["a"].mean()) < 1e-6
        assert abs(result["a"].std() - 1.0) < 0.1

    def test_transform_uses_fitted_scaler(self):
        train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        test = pd.DataFrame({"a": [10.0, 11.0]})
        _, scaler = normalize_features(train, ["a"], fit=True)
        result, _ = normalize_features(test, ["a"], scaler=scaler, fit=False)
        # Test values should be standardized relative to train mean/std
        assert result["a"].iloc[0] > 5.0  # Far from train mean

    def test_returns_scaler(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _, scaler = normalize_features(df, ["x"], fit=True)
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)


class TestImputeMissing:
    def test_median_imputation(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        result = impute_missing(df, numeric_strategy="median")
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 2.0  # median of [1, 3]

    def test_categorical_fill(self):
        df = pd.DataFrame({"cat": ["A", None, "B", None]})
        result = impute_missing(df, categorical_fill="MISSING")
        assert (result["cat"] == "MISSING").sum() == 2

    def test_no_modification_when_no_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = impute_missing(df)
        pd.testing.assert_frame_equal(df, result)

    def test_invalid_strategy_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown numeric_strategy"):
            impute_missing(df, numeric_strategy="mode")


class TestEncodeCategoricals:
    def test_basic_encoding(self):
        df = pd.DataFrame({"cat": ["A", "B", "C", "A"]})
        result, encoders = encode_categoricals(df, ["cat"], fit=True)
        assert result["cat"].dtype in [np.int64, np.int32, int]
        assert len(encoders) == 1

    def test_unseen_labels_handled(self):
        train = pd.DataFrame({"cat": ["A", "B", "C"]})
        _, encoders = encode_categoricals(train, ["cat"], fit=True)
        test = pd.DataFrame({"cat": ["A", "UNKNOWN", "B"]})
        result, _ = encode_categoricals(test, ["cat"], encoders=encoders, fit=False)
        # Should not raise; unknown mapped to first class
        assert result["cat"].isna().sum() == 0


class TestPadSequences:
    def test_short_sequences_padded(self):
        seqs = [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]
        result = pad_sequences(seqs, max_length=4, pad_value=0.0)
        assert result.shape == (2, 4)
        # Short sequence should be right-aligned (padded on left)
        assert result[0, 0] == 0.0
        assert result[0, 2] == 1.0
        assert result[0, 3] == 2.0

    def test_long_sequences_truncated(self):
        seqs = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = pad_sequences(seqs, max_length=3, truncate_side="left")
        assert result.shape == (1, 3)
        # Should keep last 3 elements
        assert list(result[0]) == [3.0, 4.0, 5.0]

    def test_exact_length_unchanged(self):
        seqs = [[1.0, 2.0, 3.0]]
        result = pad_sequences(seqs, max_length=3)
        assert result.shape == (1, 3)
        assert list(result[0]) == [1.0, 2.0, 3.0]


class TestFeaturePreprocessor:
    def test_fit_transform(self):
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, np.nan],
            "num2": [10.0, 20.0, np.nan, 40.0],
            "cat1": ["A", "B", "A", None],
        })
        preprocessor = FeaturePreprocessor(
            numeric_cols=["num1", "num2"],
            categorical_cols=["cat1"],
        )
        result = preprocessor.fit_transform(df)
        assert result["num1"].isna().sum() == 0
        assert result["cat1"].isna().sum() == 0
        assert preprocessor._fitted

    def test_transform_without_fit_raises(self):
        df = pd.DataFrame({"num1": [1.0, 2.0]})
        preprocessor = FeaturePreprocessor(numeric_cols=["num1"], categorical_cols=[])
        with pytest.raises(RuntimeError, match="must be fit"):
            preprocessor.transform(df)


class TestTemporalSplitter:
    def test_split_proportions(self):
        df = pd.DataFrame({
            "event_time": pd.date_range("2023-01-01", periods=100, freq="h"),
            "value": range(100),
        })
        splitter = TemporalSplitter(train_frac=0.7, val_frac=0.15)
        train, val, test = splitter.split(df)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_temporal_ordering_preserved(self):
        df = pd.DataFrame({
            "event_time": pd.date_range("2023-01-01", periods=50, freq="h"),
            "value": range(50),
        })
        splitter = TemporalSplitter()
        train, val, test = splitter.split(df)
        # Train max time < val min time < test min time
        assert train["event_time"].max() <= val["event_time"].min()
        assert val["event_time"].max() <= test["event_time"].min()

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            TemporalSplitter(train_frac=0.8, val_frac=0.3)
