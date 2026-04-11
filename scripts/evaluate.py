"""CLI script for evaluating trained persona and autobid models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ad_ml.config.settings import AutobidConfig, UserPersonaConfig
from ad_ml.data.dataset import CampaignBidDataset, UserBehaviorDataset
from ad_ml.evaluation.evaluator import ModelEvaluator
from ad_ml.models.autobid.model import AutobidNet
from ad_ml.models.user_persona.model import UserPersonaNet
from ad_ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ad-ml models")
    parser.add_argument("--persona-config", type=Path, default=None)
    parser.add_argument("--autobid-config", type=Path, default=None)
    parser.add_argument(
        "--persona-checkpoint",
        type=Path,
        default=Path("checkpoints/user_persona/best.pt"),
    )
    parser.add_argument(
        "--autobid-checkpoint",
        type=Path,
        default=Path("checkpoints/autobid/best.pt"),
    )
    parser.add_argument(
        "--persona-data-dir",
        type=Path,
        default=Path("data/processed/user_persona"),
    )
    parser.add_argument(
        "--autobid-data-dir",
        type=Path,
        default=Path("data/processed/autobid"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_reports"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_persona_model(checkpoint_path: Path, config: UserPersonaConfig) -> UserPersonaNet:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state_dict", state)
    # Infer dims from weight shapes
    seq_enc_w = model_state["seq_encoder.weight_ih_l0"]
    seq_feature_dim = seq_enc_w.shape[1]
    user_proj_w = model_state["user_feat_proj.0.weight"]
    user_feature_dim = user_proj_w.shape[1]
    model = UserPersonaNet(
        seq_feature_dim=seq_feature_dim,
        user_feature_dim=user_feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        num_segments=config.num_segments,
        gru_hidden_size=config.gru_hidden_size,
        gru_num_layers=config.gru_num_layers,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    )
    model.load_state_dict(model_state)
    return model


def load_autobid_model(checkpoint_path: Path, config: AutobidConfig, input_dim: int) -> AutobidNet:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state_dict", state)
    model = AutobidNet(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        cross_layers=config.cross_layers,
        dropout=config.dropout,
        min_bid=config.bid_range[0],
        max_bid=config.bid_range[1],
    )
    model.load_state_dict(model_state)
    return model


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = ModelEvaluator(device=device, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Persona evaluation ---
    if args.persona_config and args.persona_checkpoint.exists():
        logger.info("Evaluating UserPersonaNet...")
        with open(args.persona_config) as f:
            persona_cfg = UserPersonaConfig(**yaml.safe_load(f))

        test_seqs = list(
            np.load(args.persona_data_dir / "test_sequences.npy", allow_pickle=True)
        )
        test_ufeats = np.load(args.persona_data_dir / "test_user_features.npy")
        test_ds = UserBehaviorDataset(
            test_seqs, test_ufeats, max_length=persona_cfg.max_sequence_length
        )

        model = load_persona_model(args.persona_checkpoint, persona_cfg)
        metrics = evaluator.evaluate_persona(model, test_ds)
        report = evaluator.generate_report(
            metrics,
            model_name="user_persona",
            output_path=args.output_dir / "persona_eval.json",
        )
        logger.info("Persona metrics: %s", json.dumps(metrics, indent=2))

    # --- Autobid evaluation ---
    if args.autobid_config and args.autobid_checkpoint.exists():
        logger.info("Evaluating AutobidNet...")
        with open(args.autobid_config) as f:
            autobid_cfg = AutobidConfig(**yaml.safe_load(f))

        camp_feats = np.load(args.autobid_data_dir / "test_campaign_features.npy")
        ctx_feats = np.load(args.autobid_data_dir / "test_context_features.npy")
        bid_labels = np.load(args.autobid_data_dir / "test_bid_labels.npy")
        budget_path = args.autobid_data_dir / "test_budget_utils.npy"
        budget_utils = np.load(budget_path) if budget_path.exists() else None

        test_ds = CampaignBidDataset(camp_feats, ctx_feats, bid_labels, budget_utils)
        input_dim = camp_feats.shape[1] + ctx_feats.shape[1]

        model = load_autobid_model(args.autobid_checkpoint, autobid_cfg, input_dim)
        metrics = evaluator.evaluate_autobid(model, test_ds)
        report = evaluator.generate_report(
            metrics,
            model_name="autobid",
            output_path=args.output_dir / "autobid_eval.json",
        )
        logger.info("Autobid metrics: %s", json.dumps(metrics, indent=2))

    logger.info("Evaluation complete. Reports saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
