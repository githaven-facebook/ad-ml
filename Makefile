.PHONY: train-persona train-autobid evaluate serve test lint format export clean install

PYTHON := python
SRC_DIR := src
SCRIPTS_DIR := scripts
CONFIG_DIR := configs

install:
	pip install -e ".[dev]"

train-persona:
	$(PYTHON) $(SCRIPTS_DIR)/train_user_persona.py \
		--config $(CONFIG_DIR)/user_persona_config.yaml

train-autobid:
	$(PYTHON) $(SCRIPTS_DIR)/train_autobid.py \
		--config $(CONFIG_DIR)/autobid_config.yaml

evaluate:
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py \
		--persona-config $(CONFIG_DIR)/user_persona_config.yaml \
		--autobid-config $(CONFIG_DIR)/autobid_config.yaml

serve:
	$(PYTHON) -m ad_ml.serving.grpc_server \
		--port 50051 \
		--workers 4

test:
	pytest tests/ -v --cov=$(SRC_DIR)/ad_ml --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 $(SRC_DIR) $(SCRIPTS_DIR) tests
	mypy $(SRC_DIR)/ad_ml

format:
	black $(SRC_DIR) $(SCRIPTS_DIR) tests
	isort $(SRC_DIR) $(SCRIPTS_DIR) tests

export:
	$(PYTHON) $(SCRIPTS_DIR)/export_model.py \
		--format onnx \
		--output-dir exports/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

docker-build:
	docker build -t ad-ml:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

proto:
	python -m grpc_tools.protoc \
		-I src/ad_ml/serving/proto \
		--python_out=src/ad_ml/serving \
		--grpc_python_out=src/ad_ml/serving \
		src/ad_ml/serving/proto/inference.proto
