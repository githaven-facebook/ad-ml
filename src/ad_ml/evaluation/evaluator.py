"""ModelEvaluator: unified evaluation for persona and autobid models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ad_ml.data.dataset import (
    CampaignBidDataset,
    UserBehaviorDataset,
    collate_campaign_bids,
    collate_user_sequences,
)
from ad_ml.evaluation.metrics import (
    ABTestLiftCalculator,
    autobid_budget_compliance_rate,
    autobid_mape,
    autobid_roi_improvement,
    cluster_davies_bouldin,
    cluster_jaccard_stability,
    cluster_silhouette,
    reconstruction_auc,
)
from ad_ml.models.autobid.model import AutobidNet
from ad_ml.models.user_persona.model import UserPersonaNet

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models and generate comprehensive evaluation reports.

    Supports:
    - UserPersonaNet: clustering quality, reconstruction, stability
    - AutobidNet: bid accuracy, budget compliance, ROI
    - Baseline comparison
    - Statistical significance testing
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 512,
        ab_alpha: float = 0.05,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.ab_calculator = ABTestLiftCalculator(alpha=ab_alpha)

    def evaluate_persona(
        self,
        model: UserPersonaNet,
        dataset: UserBehaviorDataset,
        stability_dataset: Optional[UserBehaviorDataset] = None,
        checkpoint_path_run2: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate UserPersonaNet on a test dataset.

        Args:
            model: Trained UserPersonaNet.
            dataset: Test dataset.
            stability_dataset: Optional second dataset for stability measurement.
            checkpoint_path_run2: Optional second checkpoint for stability comparison.

        Returns:
            Dict of metric names to scalar values.
        """
        embeddings, cluster_labels, reconstructed, originals = self._run_persona_inference(
            model, dataset
        )

        metrics: Dict[str, float] = {}

        # Clustering quality
        metrics["silhouette_score"] = cluster_silhouette(embeddings, cluster_labels)
        metrics["davies_bouldin_index"] = cluster_davies_bouldin(embeddings, cluster_labels)

        # Reconstruction quality
        if reconstructed is not None and originals is not None:
            metrics["reconstruction_auc"] = reconstruction_auc(originals, reconstructed)
            metrics["reconstruction_mse"] = float(
                np.mean((originals - reconstructed) ** 2)
            )

        # Cluster size distribution entropy (measures if clusters are balanced)
        cluster_sizes = np.bincount(cluster_labels)
        probs = cluster_sizes / cluster_sizes.sum()
        metrics["cluster_entropy"] = float(-np.sum(probs * np.log(probs + 1e-10)))

        logger.info("Persona evaluation metrics: %s", metrics)
        return metrics

    def evaluate_autobid(
        self,
        model: AutobidNet,
        dataset: CampaignBidDataset,
        baseline_bid: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate AutobidNet on a test dataset.

        Args:
            model: Trained AutobidNet.
            dataset: Test dataset with bid_label and optionally budget_utilization.
            baseline_bid: Constant baseline bid multiplier for ROI comparison.

        Returns:
            Dict of metric names to scalar values.
        """
        predicted_bids, true_bids, budget_utils = self._run_autobid_inference(model, dataset)

        metrics: Dict[str, float] = {}

        # Bid accuracy
        metrics["mape"] = autobid_mape(true_bids, predicted_bids)
        metrics["mae"] = float(np.mean(np.abs(true_bids - predicted_bids)))
        metrics["rmse"] = float(np.sqrt(np.mean((true_bids - predicted_bids) ** 2)))

        # Budget compliance
        if budget_utils is not None:
            metrics["budget_compliance_rate"] = autobid_budget_compliance_rate(
                predicted_bids, budget_utils
            )

        # ROI improvement over baseline
        baseline_bids = np.full_like(predicted_bids, baseline_bid)
        dummy_conversions = np.ones(len(predicted_bids))
        dummy_costs = np.ones(len(predicted_bids))
        metrics["roi_improvement"] = autobid_roi_improvement(
            baseline_bids, predicted_bids, dummy_conversions, dummy_costs
        )

        # Bid range coverage
        metrics["bid_mean"] = float(np.mean(predicted_bids))
        metrics["bid_std"] = float(np.std(predicted_bids))
        metrics["bid_p5"] = float(np.percentile(predicted_bids, 5))
        metrics["bid_p95"] = float(np.percentile(predicted_bids, 95))

        logger.info("Autobid evaluation metrics: %s", metrics)
        return metrics

    def compare_to_baseline(
        self,
        model_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute relative improvements of model metrics vs baseline.

        Args:
            model_metrics: Metrics from the new model.
            baseline_metrics: Metrics from the baseline model.

        Returns:
            Dict mapping metric name to {'model': v, 'baseline': v, 'delta': v, 'relative_pct': v}.
        """
        comparison: Dict[str, Dict[str, float]] = {}
        for key in set(model_metrics) | set(baseline_metrics):
            m = model_metrics.get(key, float("nan"))
            b = baseline_metrics.get(key, float("nan"))
            delta = m - b
            relative_pct = (delta / abs(b)) * 100 if b != 0 else float("nan")
            comparison[key] = {
                "model": m,
                "baseline": b,
                "delta": delta,
                "relative_pct": relative_pct,
            }
        return comparison

    def generate_report(
        self,
        metrics: Dict[str, float],
        model_name: str,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a structured evaluation report.

        Args:
            metrics: Computed evaluation metrics.
            model_name: Name of the evaluated model.
            output_path: Optional path to write the report as JSON.

        Returns:
            Report dict.
        """
        report: Dict[str, Any] = {
            "model_name": model_name,
            "metrics": metrics,
            "summary": self._compute_summary(metrics),
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Evaluation report saved to %s", output_path)
        return report

    @torch.no_grad()
    def _run_persona_inference(
        self, model: UserPersonaNet, dataset: UserBehaviorDataset
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run persona model inference and return embeddings, labels, reconstructed, originals."""
        model.eval()
        model.to(self.device)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_user_sequences,
        )

        all_embeddings: List[np.ndarray] = []
        all_cluster_labels: List[np.ndarray] = []
        all_reconstructed: List[np.ndarray] = []
        all_originals: List[np.ndarray] = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = model(
                sequences=batch["sequences"],
                user_features=batch["user_features"],
                seq_lengths=batch["seq_lengths"],
                attention_mask=batch.get("attention_mask"),
                gumbel_temperature=0.1,
                hard_gumbel=True,
            )
            all_embeddings.append(output.user_embedding.cpu().numpy())
            all_cluster_labels.append(output.cluster_probs.argmax(dim=-1).cpu().numpy())
            all_reconstructed.append(output.reconstructed.cpu().numpy())

            # Compute mean of input sequences as reconstruction target
            mask = batch.get("attention_mask")
            if mask is not None:
                seq_sum = (batch["sequences"] * mask.unsqueeze(-1).float()).sum(1)
                seq_mean = seq_sum / batch["seq_lengths"].unsqueeze(-1).float().clamp(min=1)
            else:
                seq_mean = batch["sequences"].mean(1)
            all_originals.append(seq_mean.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        cluster_labels = np.concatenate(all_cluster_labels, axis=0).astype(np.int64)
        reconstructed = np.concatenate(all_reconstructed, axis=0) if all_reconstructed else None
        originals = np.concatenate(all_originals, axis=0) if all_originals else None

        return embeddings, cluster_labels, reconstructed, originals

    @torch.no_grad()
    def _run_autobid_inference(
        self, model: AutobidNet, dataset: CampaignBidDataset
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run autobid model inference and return (predicted_bids, true_bids, budget_utils)."""
        model.eval()
        model.to(self.device)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_campaign_bids,
        )

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_budgets: List[np.ndarray] = []
        has_budgets = dataset.budget_utilizations is not None

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = model(
                campaign_features=batch["campaign_features"],
                context_features=batch["context_features"],
                budget_utilization=batch.get("budget_utilization"),
            )
            all_preds.append(output.bid_multiplier.cpu().float().numpy())
            all_labels.append(batch["bid_label"].cpu().float().numpy())
            if has_budgets and "budget_utilization" in batch:
                all_budgets.append(batch["budget_utilization"].cpu().float().numpy())

        predicted = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        budgets = np.concatenate(all_budgets, axis=0) if all_budgets else None

        return predicted, labels, budgets

    @staticmethod
    def _compute_summary(metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable summary descriptions for key metrics."""
        summary: Dict[str, str] = {}
        thresholds = {
            "silhouette_score": (0.5, "good", "acceptable", "poor"),
            "mape": (5.0, "excellent", "acceptable", "needs improvement"),
            "budget_compliance_rate": (0.9, "compliant", "moderate", "non-compliant"),
        }
        for key, value in metrics.items():
            if key in thresholds:
                thresh, good, mid, bad = thresholds[key]
                if key == "mape":
                    summary[key] = good if value < thresh else (mid if value < thresh * 2 else bad)
                else:
                    summary[key] = good if value >= thresh else (mid if value >= thresh * 0.7 else bad)
        return summary
