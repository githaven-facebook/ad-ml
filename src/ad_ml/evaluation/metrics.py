"""Custom evaluation metrics for user persona and autobidding models."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User Persona Metrics
# ---------------------------------------------------------------------------


def cluster_silhouette(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    sample_size: Optional[int] = 10_000,
    random_state: int = 42,
) -> float:
    """Compute silhouette score for cluster quality.

    Args:
        embeddings: (N, D) user embedding array.
        cluster_labels: (N,) integer cluster assignment array.
        sample_size: If N > sample_size, subsample for efficiency.
        random_state: Random seed for subsampling.

    Returns:
        Silhouette score in [-1, 1]. Higher is better (1 = perfect separation).
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    num_clusters = len(np.unique(cluster_labels))
    if num_clusters < 2 or num_clusters >= n:
        logger.warning("Silhouette requires 2 <= num_clusters < n_samples, got %d / %d", num_clusters, n)
        return 0.0

    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        embeddings = embeddings[idx]
        cluster_labels = cluster_labels[idx]

    return float(silhouette_score(embeddings, cluster_labels, metric="cosine"))


def cluster_davies_bouldin(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """Compute Davies-Bouldin index (lower is better).

    Args:
        embeddings: (N, D) user embeddings.
        cluster_labels: (N,) integer cluster assignments.

    Returns:
        Davies-Bouldin index >= 0. Lower indicates better-separated clusters.
    """
    from sklearn.metrics import davies_bouldin_score

    num_clusters = len(np.unique(cluster_labels))
    if num_clusters < 2:
        return float("inf")
    return float(davies_bouldin_score(embeddings, cluster_labels))


def cluster_jaccard_stability(
    labels_run1: np.ndarray,
    labels_run2: np.ndarray,
) -> float:
    """Estimate cluster stability between two runs via average Jaccard similarity.

    For each cluster in run1, finds the best-matching cluster in run2 and
    computes the Jaccard coefficient. Returns the mean over all clusters.

    Args:
        labels_run1: (N,) cluster assignments from first model run.
        labels_run2: (N,) cluster assignments from second model run (same data).

    Returns:
        Mean Jaccard similarity in [0, 1]. Higher = more stable clustering.
    """
    clusters1 = np.unique(labels_run1)
    jaccard_scores: List[float] = []

    for c1 in clusters1:
        mask1 = set(np.where(labels_run1 == c1)[0])
        best_jaccard = 0.0
        for c2 in np.unique(labels_run2):
            mask2 = set(np.where(labels_run2 == c2)[0])
            intersection = len(mask1 & mask2)
            union = len(mask1 | mask2)
            if union > 0:
                j = intersection / union
                best_jaccard = max(best_jaccard, j)
        jaccard_scores.append(best_jaccard)

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


def reconstruction_auc(
    original_features: np.ndarray,
    reconstructed_features: np.ndarray,
    threshold_quantile: float = 0.8,
) -> float:
    """Compute reconstruction quality as AUC of anomaly detection.

    Uses reconstruction error to separate "normal" from "anomalous" users
    (high reconstruction error = anomaly). Returns AUROC.

    Args:
        original_features: (N, D) original feature matrix.
        reconstructed_features: (N, D) model-reconstructed features.
        threshold_quantile: Quantile of reconstruction errors to use as anomaly threshold.

    Returns:
        AUROC score in [0, 1] for reconstruction-based anomaly detection.
    """
    errors = np.mean((original_features - reconstructed_features) ** 2, axis=1)
    # Create binary labels: top (1-threshold_quantile) fraction = anomaly
    threshold = np.quantile(errors, threshold_quantile)
    labels = (errors >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # Degenerate case
    try:
        return float(roc_auc_score(labels, errors))
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Autobidding Metrics
# ---------------------------------------------------------------------------


def autobid_mape(
    true_bids: np.ndarray,
    predicted_bids: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Mean Absolute Percentage Error on bid price predictions.

    Args:
        true_bids: (N,) ground-truth optimal bid multipliers.
        predicted_bids: (N,) model-predicted bid multipliers.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE in [0, inf). Typical acceptable range: < 5%.
    """
    ape = np.abs((true_bids - predicted_bids) / (np.abs(true_bids) + epsilon))
    return float(100.0 * ape.mean())


def autobid_budget_compliance_rate(
    predicted_bids: np.ndarray,
    budget_utilizations: np.ndarray,
    target_utilization: float = 0.9,
    threshold: float = 1.0,
) -> float:
    """Fraction of auctions where the model correctly restrains bidding when over budget.

    A bid is "compliant" if: when budget_utilization > target_utilization,
    the predicted bid <= threshold (not aggressive).

    Args:
        predicted_bids: (N,) predicted bid multipliers.
        budget_utilizations: (N,) current budget utilization rates in [0, 1].
        target_utilization: Budget utilization threshold above which pacing should activate.
        threshold: Max acceptable bid multiplier when over budget.

    Returns:
        Compliance rate in [0, 1]. Higher is better.
    """
    over_budget_mask = budget_utilizations > target_utilization
    if over_budget_mask.sum() == 0:
        return 1.0  # No over-budget cases to evaluate
    compliant = (predicted_bids[over_budget_mask] <= threshold).sum()
    return float(compliant / over_budget_mask.sum())


def autobid_roi_improvement(
    baseline_bids: np.ndarray,
    model_bids: np.ndarray,
    conversions: np.ndarray,
    costs: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Estimate ROI improvement of model bids over a baseline bid strategy.

    Simplified model: revenue proportional to conversions, cost proportional to bid.

    Args:
        baseline_bids: (N,) baseline bid multipliers.
        model_bids: (N,) model-predicted bid multipliers.
        conversions: (N,) number of conversions per auction.
        costs: (N,) cost per auction at baseline bid of 1.0.
        epsilon: Numerical stability constant.

    Returns:
        Relative ROI improvement: (model_roi - baseline_roi) / baseline_roi.
        Positive = model outperforms baseline.
    """
    baseline_roi = (conversions.sum()) / (costs * baseline_bids + epsilon).sum()
    model_roi = (conversions.sum()) / (costs * model_bids + epsilon).sum()
    return float((model_roi - baseline_roi) / (baseline_roi + epsilon))


class ABTestLiftCalculator:
    """Compute statistical lift and significance for A/B test results.

    Uses a two-sample t-test for continuous metrics (CTR, CVR, CPA)
    and a Chi-squared test for conversion rate.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def compute_lift(
        self,
        control_metric: np.ndarray,
        treatment_metric: np.ndarray,
        metric_name: str = "metric",
    ) -> Dict[str, float]:
        """Compute lift statistics between control and treatment groups.

        Args:
            control_metric: (N_control,) metric values for control group.
            treatment_metric: (N_treatment,) metric values for treatment group.
            metric_name: Human-readable metric name for logging.

        Returns:
            Dict with keys: absolute_lift, relative_lift, p_value, is_significant,
            control_mean, treatment_mean, confidence_interval_low, confidence_interval_high.
        """
        control_mean = float(np.mean(control_metric))
        treatment_mean = float(np.mean(treatment_metric))
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / (abs(control_mean) + 1e-10)

        t_stat, p_value = stats.ttest_ind(treatment_metric, control_metric, equal_var=False)
        is_significant = bool(p_value < self.alpha)

        # 95% CI on treatment - control difference via bootstrap approximation
        pooled_se = float(
            np.sqrt(
                np.var(treatment_metric, ddof=1) / len(treatment_metric)
                + np.var(control_metric, ddof=1) / len(control_metric)
            )
        )
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci_low = absolute_lift - z * pooled_se
        ci_high = absolute_lift + z * pooled_se

        result = {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "p_value": float(p_value),
            "is_significant": float(is_significant),
            "confidence_interval_low": float(ci_low),
            "confidence_interval_high": float(ci_high),
        }
        logger.info(
            "A/B lift for %s: %.4f -> %.4f (relative=%.2f%%, p=%.4f, sig=%s)",
            metric_name, control_mean, treatment_mean,
            relative_lift * 100, p_value, is_significant,
        )
        return result
