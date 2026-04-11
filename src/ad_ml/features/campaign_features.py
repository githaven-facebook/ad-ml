"""Campaign performance feature extraction."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Smoothing constant for rate calculations (Laplace smoothing)
_SMOOTHING = 1e-6


class CampaignFeatureExtractor:
    """Extract features from historical campaign performance data.

    Features include:
    - Historical CTR, CVR, CPA over multiple time windows
    - Spend velocity (daily spend trend)
    - Budget utilization rate
    - Audience overlap score
    - Creative fatigue indicators (frequency-based)
    - Day-parting performance (by hour-of-day and day-of-week)
    """

    def __init__(
        self,
        time_windows_days: Optional[List[int]] = None,
        min_impressions: int = 100,
    ) -> None:
        self.time_windows_days = time_windows_days or [1, 7, 30]
        self.min_impressions = min_impressions

    def extract(
        self,
        campaign_logs: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Extract features for all campaigns.

        Args:
            campaign_logs: DataFrame with columns campaign_id, date, impressions,
                           clicks, conversions, spend, budget, audience_size,
                           creative_id, hour_of_day, day_of_week.
            reference_date: Cutoff date for feature computation.

        Returns:
            DataFrame indexed by campaign_id with all feature columns.
        """
        campaign_logs = campaign_logs.copy()
        campaign_logs["date"] = pd.to_datetime(campaign_logs["date"])
        ref_date = reference_date or campaign_logs["date"].max()

        feature_frames: List[pd.DataFrame] = []

        # Performance metrics per time window
        for days in self.time_windows_days:
            feature_frames.append(
                self._performance_metrics(campaign_logs, ref_date, days)
            )

        # Spend velocity
        feature_frames.append(self._spend_velocity(campaign_logs, ref_date))

        # Budget utilization
        if "budget" in campaign_logs.columns:
            feature_frames.append(self._budget_utilization(campaign_logs))

        # Creative fatigue
        if "creative_id" in campaign_logs.columns:
            feature_frames.append(self._creative_fatigue(campaign_logs))

        # Day-parting performance
        if "hour_of_day" in campaign_logs.columns:
            feature_frames.append(self._dayparting_performance(campaign_logs))

        result = feature_frames[0]
        for df in feature_frames[1:]:
            result = result.join(df, how="outer")
        return result.fillna(0.0)

    def _performance_metrics(
        self,
        logs: pd.DataFrame,
        ref_date: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:
        """Compute CTR, CVR, CPA for a given lookback window."""
        cutoff = ref_date - pd.Timedelta(days=days)
        window = logs[logs["date"] >= cutoff]
        agg = window.groupby("campaign_id").agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("spend", "sum"),
        )
        suffix = f"_{days}d"
        agg[f"ctr{suffix}"] = (agg["clicks"] + _SMOOTHING) / (agg["impressions"] + _SMOOTHING)
        agg[f"cvr{suffix}"] = (agg["conversions"] + _SMOOTHING) / (agg["clicks"] + _SMOOTHING)
        agg[f"cpa{suffix}"] = agg["spend"] / (agg["conversions"] + _SMOOTHING)
        agg[f"impressions{suffix}"] = agg["impressions"]
        agg[f"spend{suffix}"] = agg["spend"]
        return agg[[f"ctr{suffix}", f"cvr{suffix}", f"cpa{suffix}", f"impressions{suffix}", f"spend{suffix}"]]

    def _spend_velocity(
        self,
        logs: pd.DataFrame,
        ref_date: pd.Timestamp,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Compute daily spend trend (linear regression slope over recent days)."""
        cutoff = ref_date - pd.Timedelta(days=lookback_days)
        window = logs[logs["date"] >= cutoff].copy()
        window["day_idx"] = (window["date"] - cutoff).dt.days

        results: Dict[str, Dict[str, float]] = {}
        for cid, group in window.groupby("campaign_id"):
            daily = group.groupby("day_idx")["spend"].sum().reset_index()
            if len(daily) >= 2:
                x = daily["day_idx"].values.astype(np.float64)
                y = daily["spend"].values.astype(np.float64)
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = 0.0
            mean_spend = float(daily["spend"].mean())
            results[str(cid)] = {
                "spend_velocity_slope": slope,
                "spend_daily_mean": mean_spend,
            }
        return pd.DataFrame.from_dict(results, orient="index")

    def _budget_utilization(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Compute mean daily budget utilization rate."""
        logs = logs.copy()
        logs["util_rate"] = logs["spend"] / logs["budget"].replace(0, np.nan)
        agg = logs.groupby("campaign_id")["util_rate"].agg(["mean", "std"]).fillna(0.0)
        agg.columns = ["budget_util_mean", "budget_util_std"]
        agg.index = agg.index.astype(str)
        return agg

    def _creative_fatigue(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Compute creative fatigue: max frequency (impressions per unique user proxy)."""
        agg = logs.groupby("campaign_id").agg(
            total_impressions=("impressions", "sum"),
            unique_creatives=("creative_id", "nunique"),
        )
        # Fatigue proxy: impressions per creative (higher = more fatigue)
        agg["creative_fatigue"] = agg["total_impressions"] / agg["unique_creatives"].replace(0, 1)
        agg.index = agg.index.astype(str)
        return agg[["creative_fatigue"]]

    def _dayparting_performance(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Compute CTR per hour-of-day as a 24-dim feature vector."""
        pivot = logs.groupby(["campaign_id", "hour_of_day"]).agg(
            imp=("impressions", "sum"),
            clk=("clicks", "sum"),
        )
        pivot["ctr"] = (pivot["clk"] + _SMOOTHING) / (pivot["imp"] + _SMOOTHING)
        ctr_by_hour = (
            pivot["ctr"]
            .unstack(fill_value=0.0)
            .reindex(columns=list(range(24)), fill_value=0.0)
        )
        ctr_by_hour.columns = [f"daypart_ctr_h{h:02d}" for h in range(24)]
        ctr_by_hour.index = ctr_by_hour.index.astype(str)
        return ctr_by_hour
