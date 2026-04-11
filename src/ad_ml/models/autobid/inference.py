"""Autobidding real-time inference: warm-up, batched prediction, and caching."""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ad_ml.data.dataset import CampaignBidDataset, collate_campaign_bids
from ad_ml.models.autobid.model import AutobidNet

logger = logging.getLogger(__name__)


class BidPrediction(NamedTuple):
    """Result of a bid prediction call."""

    bid_multipliers: np.ndarray    # (N,) float32 in [min_bid, max_bid]
    latency_ms: float              # Inference latency in milliseconds
    cache_hits: int                # Number of results served from cache


class AutobidInference:
    """Real-time and batch inference for AutobidNet.

    Features:
    - Model warm-up on initialization
    - In-memory LRU prediction cache (keyed on feature hash)
    - Batched inference for throughput
    - Latency tracking
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_kwargs: Optional[Dict[str, object]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 256,
        cache_size: int = 10_000,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[float, float]] = {}  # key -> (value, timestamp)
        self._cache_size = cache_size

        self.model = self._load_model(checkpoint_path, model_kwargs or {})
        self.model.eval()

        # Warm up immediately
        self._warmed_up = False

    def _load_model(
        self, checkpoint_path: Path, model_kwargs: Dict[str, object]
    ) -> AutobidNet:
        state = torch.load(checkpoint_path, map_location=self.device)
        model_state = state.get("model_state_dict", state)
        model = AutobidNet(**model_kwargs)  # type: ignore[arg-type]
        model.load_state_dict(model_state)
        model.to(self.device)
        logger.info("Loaded AutobidNet from %s", checkpoint_path)
        return model

    def warmup(
        self,
        campaign_feature_dim: int,
        context_feature_dim: int,
        num_batches: int = 3,
    ) -> None:
        """Run several dummy forward passes to warm up CUDA kernels and allocator."""
        logger.info("Warming up AutobidInference (%d batches)...", num_batches)
        dummy_campaign = torch.zeros(
            self.batch_size, campaign_feature_dim, device=self.device
        )
        dummy_context = torch.zeros(
            self.batch_size, context_feature_dim, device=self.device
        )
        with torch.no_grad():
            for _ in range(num_batches):
                self.model(dummy_campaign, dummy_context)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmed_up = True
        logger.info("Warm-up complete")

    @torch.no_grad()
    def predict(
        self,
        campaign_features: np.ndarray,
        context_features: np.ndarray,
        budget_utilization: Optional[np.ndarray] = None,
    ) -> BidPrediction:
        """Run inference on a batch of auction requests.

        Args:
            campaign_features: (N, campaign_dim) float32 array.
            context_features: (N, context_dim) float32 array.
            budget_utilization: Optional (N,) float32 array with [0,1] budget fractions.

        Returns:
            BidPrediction with bid_multipliers, latency_ms, and cache_hit count.
        """
        t0 = time.perf_counter()
        n = len(campaign_features)
        result = np.zeros(n, dtype=np.float32)
        cache_hits = 0

        # Check cache for each sample
        uncached_indices: List[int] = []
        for i in range(n):
            key = self._make_cache_key(campaign_features[i], context_features[i])
            cached = self._get_cached(key)
            if cached is not None:
                result[i] = cached
                cache_hits += 1
            else:
                uncached_indices.append(i)

        # Run model for uncached samples in batches
        if uncached_indices:
            cf_batch = campaign_features[uncached_indices].astype(np.float32)
            ctx_batch = context_features[uncached_indices].astype(np.float32)
            bu_batch = (
                budget_utilization[uncached_indices].astype(np.float32)
                if budget_utilization is not None
                else None
            )

            bids = self._run_model_batched(cf_batch, ctx_batch, bu_batch)

            # Store in cache and fill result
            for j, i in enumerate(uncached_indices):
                key = self._make_cache_key(campaign_features[i], context_features[i])
                self._set_cached(key, float(bids[j]))
                result[i] = bids[j]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BidPrediction(
            bid_multipliers=result,
            latency_ms=latency_ms,
            cache_hits=cache_hits,
        )

    def _run_model_batched(
        self,
        campaign_features: np.ndarray,
        context_features: np.ndarray,
        budget_utilization: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run model inference in mini-batches and return concatenated bids."""
        n = len(campaign_features)
        all_bids: List[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            cf = torch.from_numpy(campaign_features[start:end]).to(self.device)
            ctx = torch.from_numpy(context_features[start:end]).to(self.device)
            bu: Optional[Tensor] = None
            if budget_utilization is not None:
                bu = torch.from_numpy(budget_utilization[start:end]).to(self.device)
            output = self.model(cf, ctx, bu)
            all_bids.append(output.bid_multiplier.cpu().float().numpy())

        return np.concatenate(all_bids, axis=0)

    @torch.no_grad()
    def predict_dataset(self, dataset: CampaignBidDataset) -> BidPrediction:
        """Run inference over an entire CampaignBidDataset."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_campaign_bids,
            num_workers=0,
        )
        t0 = time.perf_counter()
        all_bids: List[np.ndarray] = []
        for batch in loader:
            cf = batch["campaign_features"].to(self.device)
            ctx = batch["context_features"].to(self.device)
            bu = batch.get("budget_utilization")
            if bu is not None:
                bu = bu.to(self.device)
            output = self.model(cf, ctx, bu)
            all_bids.append(output.bid_multiplier.cpu().float().numpy())

        bids = np.concatenate(all_bids, axis=0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return BidPrediction(bid_multipliers=bids, latency_ms=latency_ms, cache_hits=0)

    @staticmethod
    def _make_cache_key(campaign_feat: np.ndarray, context_feat: np.ndarray) -> str:
        combined = np.concatenate([campaign_feat.ravel(), context_feat.ravel()])
        return hashlib.md5(combined.tobytes(), usedforsecurity=False).hexdigest()

    def _get_cached(self, key: str) -> Optional[float]:
        if key not in self._cache:
            return None
        value, ts = self._cache[key]
        if time.time() - ts > self.cache_ttl:
            del self._cache[key]
            return None
        return value

    def _set_cached(self, key: str, value: float) -> None:
        if len(self._cache) >= self._cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time())

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
