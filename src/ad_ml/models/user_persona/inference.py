"""User persona inference pipeline: batch prediction and feature store export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ad_ml.data.dataset import UserBehaviorDataset, collate_user_sequences
from ad_ml.models.user_persona.model import UserPersonaNet

logger = logging.getLogger(__name__)


class PersonaPrediction(NamedTuple):
    """Inference output for a batch of users."""

    user_ids: List[str]
    embeddings: np.ndarray         # (N, embedding_dim) float32
    cluster_assignments: np.ndarray  # (N,) int64 — argmax cluster
    cluster_probs: np.ndarray       # (N, num_segments) float32


class PersonaInference:
    """Load a trained UserPersonaNet and run batch inference.

    Args:
        checkpoint_path: Path to saved .pt checkpoint.
        model_kwargs: Constructor kwargs for UserPersonaNet (if no checkpoint metadata).
        device: Compute device.
        batch_size: Inference batch size.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_kwargs: Optional[Dict[str, object]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 512,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = self._load_model(checkpoint_path, model_kwargs or {})
        self.model.eval()

    def _load_model(
        self, checkpoint_path: Path, model_kwargs: Dict[str, object]
    ) -> UserPersonaNet:
        state = torch.load(checkpoint_path, map_location=self.device)
        model_state = state.get("model_state_dict", state)

        # Infer dims from checkpoint if not provided
        if "seq_feature_dim" not in model_kwargs:
            # Try to infer from weight shapes
            seq_enc_weight = model_state.get("seq_encoder.weight_ih_l0")
            if seq_enc_weight is not None:
                model_kwargs["seq_feature_dim"] = seq_enc_weight.shape[1]

        model = UserPersonaNet(**model_kwargs)  # type: ignore[arg-type]
        model.load_state_dict(model_state)
        model.to(self.device)
        logger.info("Loaded PersonaNet from %s", checkpoint_path)
        return model

    @torch.no_grad()
    def predict_batch(
        self,
        dataset: UserBehaviorDataset,
        user_ids: Optional[List[str]] = None,
    ) -> PersonaPrediction:
        """Run inference over an entire UserBehaviorDataset.

        Args:
            dataset: Dataset to run inference on.
            user_ids: Optional user ID list aligned with dataset indices.

        Returns:
            PersonaPrediction with embeddings and cluster assignments.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_user_sequences,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        all_embeddings: List[np.ndarray] = []
        all_cluster_probs: List[np.ndarray] = []

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            output = self.model(
                sequences=batch["sequences"],
                user_features=batch["user_features"],
                seq_lengths=batch["seq_lengths"],
                attention_mask=batch.get("attention_mask"),
                gumbel_temperature=0.1,  # Low temp for deterministic assignments
                hard_gumbel=True,
            )
            all_embeddings.append(output.user_embedding.cpu().float().numpy())
            all_cluster_probs.append(output.cluster_probs.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        cluster_probs = np.concatenate(all_cluster_probs, axis=0)
        cluster_assignments = cluster_probs.argmax(axis=1).astype(np.int64)

        ids = user_ids or [str(i) for i in range(len(dataset))]

        return PersonaPrediction(
            user_ids=ids,
            embeddings=embeddings,
            cluster_assignments=cluster_assignments,
            cluster_probs=cluster_probs,
        )

    def export_to_redis(
        self,
        predictions: PersonaPrediction,
        redis_client: object,
        key_prefix: str = "persona:",
        ttl_seconds: int = 86400,
    ) -> None:
        """Write user embeddings to Redis as binary blobs.

        Args:
            predictions: Output from predict_batch().
            redis_client: redis.Redis client instance.
            key_prefix: Key prefix for Redis entries.
            ttl_seconds: TTL for each key.
        """
        import redis  # type: ignore[import]

        pipe = redis_client.pipeline(transaction=False)  # type: ignore[union-attr]
        for uid, emb, cluster in zip(
            predictions.user_ids,
            predictions.embeddings,
            predictions.cluster_assignments,
        ):
            key = f"{key_prefix}{uid}"
            value = np.concatenate([emb, [float(cluster)]]).astype(np.float32).tobytes()
            pipe.setex(key, ttl_seconds, value)
        pipe.execute()
        logger.info(
            "Exported %d user embeddings to Redis with prefix '%s'",
            len(predictions.user_ids),
            key_prefix,
        )

    def export_to_parquet(
        self, predictions: PersonaPrediction, output_path: Path
    ) -> None:
        """Save predictions to a Parquet file for batch downstream use."""
        import pandas as pd

        emb_cols = {
            f"emb_{i}": predictions.embeddings[:, i]
            for i in range(predictions.embeddings.shape[1])
        }
        cluster_cols = {
            f"cluster_prob_{k}": predictions.cluster_probs[:, k]
            for k in range(predictions.cluster_probs.shape[1])
        }
        df = pd.DataFrame(
            {
                "user_id": predictions.user_ids,
                "cluster_assignment": predictions.cluster_assignments,
                **emb_cols,
                **cluster_cols,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Exported %d predictions to %s", len(df), output_path)

    def warmup(self, seq_feature_dim: int, user_feature_dim: int) -> None:
        """Run a dummy forward pass to warm up CUDA kernels."""
        dummy_seq = torch.zeros(1, 10, seq_feature_dim, device=self.device)
        dummy_feat = torch.zeros(1, user_feature_dim, device=self.device)
        dummy_len = torch.tensor([10], device=self.device)
        with torch.no_grad():
            self.model(dummy_seq, dummy_feat, dummy_len)
        logger.debug("PersonaInference warmed up")
