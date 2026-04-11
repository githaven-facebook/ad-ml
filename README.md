# ad-ml

Facebook Ad System ML Models - User Persona Clustering and Autobidding.

## Overview

Two core ML systems:

1. **User Persona Model** - Clusters users into behavioral segments using sequential event data from S3. Outputs dense user embedding vectors and soft persona assignments used for audience targeting.

2. **Autobidding Model** - Predicts optimal bid multipliers for ad campaigns in real time. Uses campaign performance history, contextual signals, and budget constraints to maximize campaign objectives.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train user persona model
make train-persona

# Train autobidding model
make train-autobid

# Evaluate models
make evaluate

# Run tests
make test

# Start gRPC serving
make serve
```

## Repository Structure

```
src/ad_ml/        - Core library
  config/         - Pydantic settings and YAML configs
  data/           - S3 loading, preprocessing, PyTorch datasets
  features/       - Feature engineering pipelines
  models/         - Model architectures and training loops
  evaluation/     - Metrics and evaluation framework
  serving/        - gRPC inference server
  utils/          - Logging and experiment tracking
scripts/          - CLI training and export scripts
configs/          - YAML configuration files
tests/            - Unit and integration tests
```

## Model Architectures

### User Persona Network

- Categorical embedding layers for user attributes
- GRU encoder for sequential action history
- Multi-head self-attention for temporal importance weighting
- MLP projection head outputting 128-dim user embedding
- Auxiliary clustering head with Gumbel-Softmax for 32 persona segments
- Loss: reconstruction + KL divergence clustering + contrastive

### Autobidding Network (DCN-V2)

- Cross network for explicit high-order feature interactions
- Deep MLP for implicit feature patterns
- Budget pacing constraint layer
- Output: bid multiplier scaled to configured [min, max] range
- Loss: MSE on optimal bid + constraint penalty + entropy regularization

## Training

```bash
# User persona with custom config
python scripts/train_user_persona.py \
  --config configs/user_persona_config.yaml \
  --experiment-name persona-v2

# Autobidding model
python scripts/train_autobid.py \
  --config configs/autobid_config.yaml \
  --experiment-name autobid-v3
```

## Serving

The gRPC inference server loads both models and exposes:

- `Predict` - single sample inference
- `BatchPredict` - batched inference
- `GetModelInfo` - model metadata and health

```bash
# Start server
python -m ad_ml.serving.grpc_server --port 50051

# Or via Docker
docker-compose up serving
```

## Experiment Tracking

MLflow is used for experiment tracking. View runs at `http://localhost:5000`.

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

## Deployment

CI/CD via GitHub Actions:

- `.github/workflows/ci.yml` - lint, type-check, test on every PR
- `.github/workflows/cd.yml` - build Docker image and deploy on merge to main
