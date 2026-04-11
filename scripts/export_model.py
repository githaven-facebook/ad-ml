"""Export trained models to ONNX and TorchScript formats."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from ad_ml.config.settings import AutobidConfig, UserPersonaConfig
from ad_ml.models.autobid.model import AutobidNet
from ad_ml.models.user_persona.model import UserPersonaNet
from ad_ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ad-ml models to ONNX or TorchScript")
    parser.add_argument(
        "--model",
        type=str,
        choices=["user_persona", "autobid", "both"],
        default="both",
        help="Which model to export",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "both"],
        default="both",
    )
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
        "--persona-config",
        type=Path,
        default=Path("configs/user_persona_config.yaml"),
    )
    parser.add_argument(
        "--autobid-config",
        type=Path,
        default=Path("configs/autobid_config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def export_persona_onnx(
    model: UserPersonaNet,
    output_path: Path,
    opset_version: int = 17,
    seq_len: int = 32,
    seq_feature_dim: int = 64,
    user_feature_dim: int = 32,
) -> None:
    """Export UserPersonaNet to ONNX."""
    model.eval()
    dummy_seq = torch.zeros(1, seq_len, seq_feature_dim)
    dummy_user_feat = torch.zeros(1, user_feature_dim)
    dummy_lengths = torch.tensor([seq_len])

    torch.onnx.export(
        model,
        (dummy_seq, dummy_user_feat, dummy_lengths),
        str(output_path),
        opset_version=opset_version,
        input_names=["sequences", "user_features", "seq_lengths"],
        output_names=[
            "user_embedding",
            "cluster_logits",
            "cluster_probs",
            "reconstructed",
            "attention_weights",
        ],
        dynamic_axes={
            "sequences": {0: "batch_size", 1: "seq_len"},
            "user_features": {0: "batch_size"},
            "seq_lengths": {0: "batch_size"},
            "user_embedding": {0: "batch_size"},
            "cluster_probs": {0: "batch_size"},
        },
    )
    logger.info("Exported UserPersonaNet to ONNX: %s", output_path)


def export_autobid_onnx(
    model: AutobidNet,
    output_path: Path,
    opset_version: int = 17,
    campaign_dim: int = 64,
    context_dim: int = 64,
) -> None:
    """Export AutobidNet to ONNX."""
    model.eval()
    dummy_campaign = torch.zeros(1, campaign_dim)
    dummy_context = torch.zeros(1, context_dim)

    torch.onnx.export(
        model,
        (dummy_campaign, dummy_context),
        str(output_path),
        opset_version=opset_version,
        input_names=["campaign_features", "context_features"],
        output_names=["bid_multiplier", "bid_raw", "cross_features", "deep_features"],
        dynamic_axes={
            "campaign_features": {0: "batch_size"},
            "context_features": {0: "batch_size"},
            "bid_multiplier": {0: "batch_size"},
        },
    )
    logger.info("Exported AutobidNet to ONNX: %s", output_path)


def export_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    model_name: str,
) -> None:
    """Export a model to TorchScript via tracing."""
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(output_path))
        logger.info("Exported %s to TorchScript: %s", model_name, output_path)
    except Exception as e:
        logger.warning("torch.jit.script failed for %s: %s. Skipping.", model_name, e)


def load_persona_from_checkpoint(
    checkpoint_path: Path, config: UserPersonaConfig
) -> Tuple[UserPersonaNet, int, int]:
    """Load model and infer feature dims from checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state_dict", state)
    seq_feature_dim = model_state["seq_encoder.weight_ih_l0"].shape[1]
    user_feature_dim = model_state["user_feat_proj.0.weight"].shape[1]

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
    return model, seq_feature_dim, user_feature_dim


def load_autobid_from_checkpoint(
    checkpoint_path: Path, config: AutobidConfig
) -> Tuple[AutobidNet, int]:
    """Load AutobidNet from checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state_dict", state)
    # Infer input_dim from first cross layer weight shape
    cross_weight = model_state["cross_network.layers.0.weight"]
    input_dim = cross_weight.shape[0]

    model = AutobidNet(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        cross_layers=config.cross_layers,
        dropout=config.dropout,
        min_bid=config.bid_range[0],
        max_bid=config.bid_range[1],
    )
    model.load_state_dict(model_state)
    return model, input_dim


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export User Persona model
    if args.model in ("user_persona", "both") and args.persona_checkpoint.exists():
        with open(args.persona_config) as f:
            persona_cfg = UserPersonaConfig(**yaml.safe_load(f))

        model, seq_dim, user_dim = load_persona_from_checkpoint(
            args.persona_checkpoint, persona_cfg
        )

        if args.format in ("onnx", "both"):
            onnx_path = args.output_dir / "user_persona.onnx"
            export_persona_onnx(
                model, onnx_path, opset_version=args.opset_version,
                seq_feature_dim=seq_dim, user_feature_dim=user_dim,
            )

        if args.format in ("torchscript", "both"):
            ts_path = args.output_dir / "user_persona.pt"
            export_torchscript(model, ts_path, "UserPersonaNet")

    # Export Autobid model
    if args.model in ("autobid", "both") and args.autobid_checkpoint.exists():
        with open(args.autobid_config) as f:
            autobid_cfg = AutobidConfig(**yaml.safe_load(f))

        model_ab, input_dim = load_autobid_from_checkpoint(args.autobid_checkpoint, autobid_cfg)
        # Infer campaign vs context split (assume 50/50 if not stored)
        half_dim = input_dim // 2

        if args.format in ("onnx", "both"):
            onnx_path = args.output_dir / "autobid.onnx"
            export_autobid_onnx(
                model_ab, onnx_path, opset_version=args.opset_version,
                campaign_dim=half_dim, context_dim=input_dim - half_dim,
            )

        if args.format in ("torchscript", "both"):
            ts_path = args.output_dir / "autobid.pt"
            export_torchscript(model_ab, ts_path, "AutobidNet")

    logger.info("Export complete. Files written to %s", args.output_dir)


if __name__ == "__main__":
    main()
