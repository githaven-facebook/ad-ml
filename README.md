# ad-ml

Facebook Ad System ML Models — User Persona Clustering and Autobidding.

## Overview

Two core ML systems powering ad targeting and bidding:

1. **User Persona Model** — Segments users into behavioral clusters using sequential event data from S3. Produces 128-dim user embedding vectors and soft persona assignments (32 segments) for audience targeting downstream.

2. **Autobidding Model** — Predicts optimal bid multipliers for ad campaigns in real time. Uses a Deep & Cross Network V2 (DCN-V2) architecture with campaign performance history, contextual signals, and a budget pacing constraint layer. Output: bid multiplier in [0.5, 3.0].

## Architecture

### User Persona Network

```
User Events (sequences)         User Attributes
      │                                │
  GRU Encoder                   Linear Projection
  (2-layer, 256-dim)             (→ 256-dim)
      │                                │
  Multi-Head Self-Attention       ─────┘
  (8 heads, temporal weighting)
      │
  Concat + MLP [256→128→64→128]
      │
  L2-Normalized User Embedding (128-dim)
      │                    │
  Clustering Head      Reconstruction Head
  (Gumbel-Softmax)    (MSE auxiliary loss)
  → 32 persona clusters
```

Loss: `L = reconstruction + 0.1 * KL_clustering + 0.05 * InfoNCE_contrastive`

### Autobidding Network (DCN-V2)

```
Campaign Features + Context Features
              │
       BatchNorm (input normalization)
          ┌───┴───┐
   Cross Network   Deep Network (MLP)
   (3 cross layers) [512→256→128]
   (explicit interactions) (implicit patterns)
          └───┬───┘
           Concat
           Linear → Linear → bid_raw
              │
   BudgetPacingConstraint (learnable pacing)
              │
   Sigmoid scaling → bid_multiplier ∈ [0.5, 3.0]
```

Loss: `L = Huber(bid_pred, bid_target) + 0.5 * budget_constraint_penalty + 0.01 * entropy_reg`

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train
make train-persona
make train-autobid

# Evaluate
make evaluate

# Serve (gRPC on port 50051)
make serve

# Test
make test

# Lint + type check
make lint
```

## Training

### User Persona

```bash
python scripts/train_user_persona.py \
  --config configs/user_persona_config.yaml \
  --data-dir data/processed/user_persona \
  --experiment-name persona-v2 \
  --run-name run-001

# Distributed training (multi-GPU)
python scripts/train_user_persona.py \
  --config configs/user_persona_config.yaml \
  --distributed
```

Expected data layout:
```
data/processed/user_persona/
  train_sequences.npy        # list of (seq_len, feature_dim) arrays
  train_user_features.npy    # (N, user_feature_dim)
  val_sequences.npy
  val_user_features.npy
  test_sequences.npy
  test_user_features.npy
```

### Autobidding

```bash
python scripts/train_autobid.py \
  --config configs/autobid_config.yaml \
  --data-dir data/processed/autobid \
  --experiment-name autobid-v3
```

Expected data layout:
```
data/processed/autobid/
  train_campaign_features.npy  # (N, campaign_dim)
  train_context_features.npy   # (N, context_dim)
  train_bid_labels.npy         # (N,) optimal bid multipliers
  train_budget_utils.npy       # (N,) budget utilization rates (optional)
  val_*  /  test_*             # same structure
```

## Evaluation

```bash
python scripts/evaluate.py \
  --persona-config configs/user_persona_config.yaml \
  --autobid-config configs/autobid_config.yaml \
  --persona-checkpoint checkpoints/user_persona/best.pt \
  --autobid-checkpoint checkpoints/autobid/best.pt \
  --output-dir evaluation_reports/
```

Metrics reported:

| Model | Metric | Target |
|---|---|---|
| Persona | Silhouette score | > 0.5 |
| Persona | Davies-Bouldin index | < 1.0 |
| Persona | Cluster Jaccard stability | > 0.7 |
| Autobid | MAPE on bid price | < 5% |
| Autobid | Budget compliance rate | > 90% |
| Autobid | ROI improvement vs baseline | > 0% |

## Serving

The gRPC inference server exposes four RPCs:

| RPC | Description |
|---|---|
| `Predict` | Single-sample inference |
| `BatchPredict` | Batched inference for throughput clients |
| `GetModelInfo` | Model metadata and lifetime stats |
| `HealthCheck` | Readiness and health status |

```bash
# Start server
python -m ad_ml.serving.grpc_server \
  --persona-checkpoint checkpoints/user_persona/best.pt \
  --autobid-checkpoint checkpoints/autobid/best.pt \
  --port 50051 \
  --workers 8

# Or via Docker Compose
docker-compose up serving
```

Generate proto stubs (required before first run):
```bash
make proto
```

## Model Export

```bash
# Export to ONNX and TorchScript
python scripts/export_model.py \
  --model both \
  --format both \
  --output-dir exports/

# ONNX only
python scripts/export_model.py --format onnx
```

## Experiment Tracking

MLflow is used for all experiment tracking. Runs are logged with:
- All hyperparameters
- Per-epoch train/val loss
- Best checkpoint artifacts
- Model registry integration

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Or via Docker Compose
docker-compose up mlflow
# Open http://localhost:5000
```

## Docker Deployment

```bash
# Build images
docker-compose build

# Start all services (serving + MLflow + Redis)
docker-compose up -d serving mlflow redis

# Start training (GPU required)
docker-compose run --rm train-persona
docker-compose run --rm train-autobid

# Stop all
docker-compose down
```

## CI/CD

- **`.github/workflows/ci.yml`** — Runs on every PR: lint (flake8, black, isort), type-check (mypy), unit tests, integration tests, model smoke test.
- **`.github/workflows/cd.yml`** — Runs on merge to main: builds and pushes training + serving Docker images to GHCR, deploys serving container to production.

## Configuration

Key config files:

| File | Purpose |
|---|---|
| `configs/user_persona_config.yaml` | Persona model hyperparameters |
| `configs/autobid_config.yaml` | Autobid model hyperparameters |
| `src/ad_ml/config/settings.py` | Pydantic settings (S3, serving, MLflow) |

Environment variable overrides use prefix `AD_ML__` with double-underscore nesting:
```bash
export AD_ML__SERVING__PORT=50052
export AD_ML__S3__BUCKET=my-bucket
```

## Project Structure

```
src/ad_ml/
  config/         Pydantic settings and YAML configs
  data/           S3 loader, preprocessing, PyTorch datasets
  features/       User, campaign, and context feature extractors
  models/
    user_persona/ UserPersonaNet + PersonaTrainer + PersonaInference
    autobid/      AutobidNet + AutobidTrainer + AutobidInference
  evaluation/     Metrics (silhouette, MAPE, compliance) and ModelEvaluator
  serving/        gRPC server + Protobuf definitions
  utils/          Structured logging (structlog) + MLflow wrapper
scripts/          CLI training, evaluation, and export scripts
configs/          YAML training configuration files
tests/
  unit/           Model forward pass, loss, preprocessing, metrics
  integration/    End-to-end training pipeline tests
.github/workflows/ CI and CD pipelines
```
