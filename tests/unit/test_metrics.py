"""Unit tests for custom evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ad_ml.evaluation.metrics import (
    ABTestLiftCalculator,
    autobid_budget_compliance_rate,
    autobid_mape,
    autobid_roi_improvement,
    cluster_jaccard_stability,
    cluster_silhouette,
    reconstruction_auc,
)


class TestClusterSilhouette:
    def test_well_separated_clusters(self):
        rng = np.random.default_rng(42)
        # Two clearly separated clusters
        c1 = rng.normal([0.0, 0.0], 0.1, size=(50, 2))
        c2 = rng.normal([10.0, 10.0], 0.1, size=(50, 2))
        embeddings = np.vstack([c1, c2]).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50)
        score = cluster_silhouette(embeddings, labels)
        assert score > 0.5, f"Expected high silhouette score for well-separated clusters, got {score}"

    def test_single_cluster_returns_zero(self):
        embeddings = np.random.rand(20, 8).astype(np.float32)
        labels = np.zeros(20, dtype=np.int64)
        score = cluster_silhouette(embeddings, labels)
        assert score == 0.0

    def test_range_is_valid(self):
        rng = np.random.default_rng(0)
        embeddings = rng.random((100, 16)).astype(np.float32)
        labels = rng.integers(0, 4, size=100)
        score = cluster_silhouette(embeddings, labels)
        assert -1.0 <= score <= 1.0


class TestClusterJaccardStability:
    def test_identical_assignments_score_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        score = cluster_jaccard_stability(labels, labels)
        assert abs(score - 1.0) < 1e-6

    def test_completely_different_assignments(self):
        labels1 = np.array([0, 0, 0, 1, 1, 1])
        labels2 = np.array([2, 2, 3, 3, 4, 4])
        score = cluster_jaccard_stability(labels1, labels2)
        assert 0.0 <= score <= 1.0

    def test_output_in_range(self):
        rng = np.random.default_rng(7)
        labels1 = rng.integers(0, 5, size=100)
        labels2 = rng.integers(0, 5, size=100)
        score = cluster_jaccard_stability(labels1, labels2)
        assert 0.0 <= score <= 1.0


class TestReconstructionAuc:
    def test_perfect_reconstruction_returns_low_auc(self):
        features = np.random.rand(100, 16).astype(np.float32)
        score = reconstruction_auc(features, features)
        # Perfect reconstruction -> all errors near zero -> poor anomaly detector
        assert score <= 0.6

    def test_noisy_reconstruction_returns_high_auc(self):
        features = np.random.rand(100, 16).astype(np.float32)
        # Add large noise to top 20% to create clear anomalies
        corrupted = features.copy()
        corrupted[:20] += 100.0
        score = reconstruction_auc(features, corrupted)
        assert score >= 0.7


class TestAutobidMape:
    def test_zero_error(self):
        bids = np.array([1.0, 1.5, 2.0])
        mape = autobid_mape(bids, bids)
        assert abs(mape) < 1e-5

    def test_known_error(self):
        true_bids = np.array([1.0, 2.0])
        pred_bids = np.array([1.1, 2.2])  # 10% error each
        mape = autobid_mape(true_bids, pred_bids)
        assert abs(mape - 10.0) < 0.5

    def test_output_is_percentage(self):
        true_bids = np.array([1.0, 2.0, 3.0])
        pred_bids = np.array([1.5, 2.5, 3.5])
        mape = autobid_mape(true_bids, pred_bids)
        # All are < 100% off, so MAPE should be < 50%
        assert 0 <= mape < 50


class TestAutobidBudgetCompliance:
    def test_fully_compliant(self):
        # All bids <= 1.0 when over budget
        bids = np.full(10, 0.8)
        budgets = np.full(10, 0.95)  # All over 0.9 threshold
        rate = autobid_budget_compliance_rate(bids, budgets)
        assert rate == 1.0

    def test_fully_non_compliant(self):
        # All bids > 1.0 when over budget
        bids = np.full(10, 2.0)
        budgets = np.full(10, 0.95)
        rate = autobid_budget_compliance_rate(bids, budgets)
        assert rate == 0.0

    def test_no_over_budget_returns_one(self):
        bids = np.random.rand(10) * 3.0
        budgets = np.full(10, 0.5)  # All under threshold
        rate = autobid_budget_compliance_rate(bids, budgets)
        assert rate == 1.0


class TestAutobidRoiImprovement:
    def test_no_improvement_at_baseline(self):
        bids = np.ones(10)
        improvement = autobid_roi_improvement(bids, bids, np.ones(10), np.ones(10))
        assert abs(improvement) < 1e-5

    def test_lower_bids_improve_roi(self):
        baseline = np.full(10, 1.5)
        model_bids = np.full(10, 1.0)  # More conservative bids -> lower cost -> better ROI
        improvement = autobid_roi_improvement(
            baseline, model_bids, np.ones(10), np.ones(10)
        )
        assert improvement > 0, "Lower bids should improve ROI"


class TestABTestLiftCalculator:
    def test_significant_lift_detected(self):
        rng = np.random.default_rng(42)
        control = rng.normal(0.05, 0.01, size=1000)
        treatment = rng.normal(0.06, 0.01, size=1000)
        calc = ABTestLiftCalculator(alpha=0.05)
        result = calc.compute_lift(control, treatment)
        assert result["is_significant"] == 1.0
        assert result["absolute_lift"] > 0

    def test_no_significant_lift(self):
        rng = np.random.default_rng(99)
        control = rng.normal(0.05, 0.01, size=50)
        treatment = rng.normal(0.05, 0.01, size=50)
        calc = ABTestLiftCalculator(alpha=0.05)
        result = calc.compute_lift(control, treatment)
        assert "p_value" in result
        assert "relative_lift" in result

    def test_output_keys(self):
        calc = ABTestLiftCalculator()
        result = calc.compute_lift(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        expected_keys = {
            "control_mean", "treatment_mean", "absolute_lift", "relative_lift",
            "p_value", "is_significant", "confidence_interval_low", "confidence_interval_high"
        }
        assert expected_keys.issubset(result.keys())
