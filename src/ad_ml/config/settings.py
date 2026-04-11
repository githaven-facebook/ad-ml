"""Pydantic settings for ad-ml service configuration."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class S3Config(BaseSettings):
    """S3 data source configuration."""

    bucket: str = Field(default="fb-ad-ml-data", description="S3 bucket name")
    event_prefix: str = Field(default="events/", description="Prefix for user event data")
    campaign_prefix: str = Field(default="campaigns/", description="Prefix for campaign data")
    region: str = Field(default="us-east-1", description="AWS region")
    endpoint_url: Optional[str] = Field(default=None, description="Custom S3 endpoint (for testing)")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")

    model_config = {"env_prefix": "S3_"}


class UserPersonaConfig(BaseSettings):
    """User persona model hyperparameters."""

    embedding_dim: int = Field(default=128, description="Output user embedding dimension")
    hidden_dims: List[int] = Field(
        default=[256, 128, 64], description="MLP hidden layer dimensions"
    )
    num_segments: int = Field(default=32, description="Number of persona clusters")
    gru_hidden_size: int = Field(default=256, description="GRU hidden state size")
    gru_num_layers: int = Field(default=2, description="Number of GRU layers")
    attention_heads: int = Field(default=8, description="Number of self-attention heads")
    dropout: float = Field(default=0.1, description="Dropout probability")
    max_sequence_length: int = Field(default=512, description="Maximum user action sequence length")

    # Training hyperparams
    learning_rate: float = Field(default=0.001, description="Initial learning rate")
    batch_size: int = Field(default=512, description="Training batch size")
    epochs: int = Field(default=50, description="Maximum training epochs")
    warmup_steps: int = Field(default=1000, description="LR warmup steps")
    grad_clip_norm: float = Field(default=1.0, description="Gradient clipping max norm")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience epochs")

    # Loss weights
    reconstruction_weight: float = Field(default=1.0, description="Reconstruction loss weight")
    clustering_weight: float = Field(default=0.1, description="Clustering KL loss weight")
    contrastive_weight: float = Field(default=0.05, description="Contrastive loss weight")
    contrastive_temperature: float = Field(default=0.07, description="Contrastive loss temperature")

    model_config = {"env_prefix": "PERSONA_"}


class AutobidConfig(BaseSettings):
    """Autobidding model hyperparameters."""

    input_dim: int = Field(default=128, description="Input feature dimension")
    hidden_dims: List[int] = Field(
        default=[512, 256, 128], description="Deep network hidden dimensions"
    )
    cross_layers: int = Field(default=3, description="Number of DCN-V2 cross layers")
    dropout: float = Field(default=0.1, description="Dropout probability")

    # Bid constraints
    bid_range: Tuple[float, float] = Field(
        default=(0.5, 3.0), description="Min and max bid multiplier"
    )

    # Training hyperparams
    learning_rate: float = Field(default=0.0005, description="Initial learning rate")
    batch_size: int = Field(default=1024, description="Training batch size")
    epochs: int = Field(default=100, description="Maximum training epochs")
    warmup_steps: int = Field(default=2000, description="LR warmup steps")
    grad_clip_norm: float = Field(default=1.0, description="Gradient clipping max norm")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience epochs")

    # Replay buffer
    replay_buffer_size: int = Field(default=100_000, description="Online learning replay buffer size")
    replay_batch_size: int = Field(default=256, description="Replay buffer sample size")

    # Loss weights
    constraint_penalty_weight: float = Field(default=0.5, description="Budget constraint penalty weight")
    entropy_weight: float = Field(default=0.01, description="Exploration entropy regularization weight")

    @field_validator("bid_range", mode="before")
    @classmethod
    def validate_bid_range(cls, v: object) -> Tuple[float, float]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            if lo >= hi:
                raise ValueError("bid_range min must be less than max")
            return (lo, hi)
        raise ValueError("bid_range must be a 2-element list or tuple")

    model_config = {"env_prefix": "AUTOBID_"}


class ServingConfig(BaseSettings):
    """gRPC serving configuration."""

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=50051, description="gRPC port")
    max_workers: int = Field(default=8, description="Thread pool workers")
    max_message_length: int = Field(
        default=10 * 1024 * 1024, description="Max gRPC message size (bytes)"
    )
    persona_model_path: str = Field(
        default="models/user_persona/latest.pt", description="Path to persona model checkpoint"
    )
    autobid_model_path: str = Field(
        default="models/autobid/latest.pt", description="Path to autobid model checkpoint"
    )
    warmup_batch_size: int = Field(default=32, description="Warm-up batch size on startup")
    prediction_cache_ttl: int = Field(default=300, description="Prediction cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL for caching")

    model_config = {"env_prefix": "SERVING_"}


class MLflowConfig(BaseSettings):
    """MLflow experiment tracking configuration."""

    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    artifact_root: str = Field(default="s3://fb-ad-ml-mlflow/artifacts", description="Artifact storage root")
    persona_experiment: str = Field(default="user-persona", description="Persona experiment name")
    autobid_experiment: str = Field(default="autobid", description="Autobid experiment name")
    registry_uri: Optional[str] = Field(default=None, description="Model registry URI (defaults to tracking_uri)")

    model_config = {"env_prefix": "MLFLOW_"}


class Settings(BaseSettings):
    """Top-level application settings."""

    s3: S3Config = Field(default_factory=S3Config)
    user_persona: UserPersonaConfig = Field(default_factory=UserPersonaConfig)
    autobid: AutobidConfig = Field(default_factory=AutobidConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Runtime
    device: str = Field(default="cuda", description="Compute device: cuda or cpu")
    seed: int = Field(default=42, description="Global random seed")
    num_workers: int = Field(default=4, description="DataLoader worker count")
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {"env_prefix": "AD_ML_", "env_nested_delimiter": "__"}


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings()
