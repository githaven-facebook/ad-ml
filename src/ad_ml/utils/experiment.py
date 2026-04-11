"""MLflow experiment tracking wrapper with model registry integration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow for experiment tracking, artifact management, and model registry.

    Provides a simplified interface for:
    - Creating and managing experiments
    - Logging parameters, metrics, and artifacts
    - Registering models to the model registry
    - Comparing runs across experiments
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "default",
        artifact_location: Optional[str] = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment_id = self._get_or_create_experiment(experiment_name, artifact_location)
        self._active_run_id: Optional[str] = None

    def _get_or_create_experiment(
        self, name: str, artifact_location: Optional[str]
    ) -> str:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        kwargs: Dict[str, Any] = {"name": name}
        if artifact_location:
            kwargs["artifact_location"] = artifact_location
        experiment_id = mlflow.create_experiment(**kwargs)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
        return experiment_id

    @contextmanager
    def run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Generator["ExperimentTracker", None, None]:
        """Context manager that starts and ends a MLflow run.

        Example:
            with tracker.run(run_name="train-v2") as run:
                run.log_params({"lr": 0.001})
                run.log_metric("loss", 0.42, step=1)
        """
        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
        ) as active_run:
            self._active_run_id = active_run.info.run_id
            try:
                yield self
            finally:
                self._active_run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dict of hyperparameters to the active run."""
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """Log a single scalar metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an MLflow artifact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log a Python dict as a JSON artifact."""
        mlflow.log_dict(data, artifact_file)

    def log_pytorch_model(
        self,
        model: Any,
        artifact_path: str = "model",
        conda_env: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None,
    ) -> None:
        """Log a PyTorch model as an MLflow artifact."""
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            signature=signature,
        )
        logger.info("Logged PyTorch model to MLflow artifact path '%s'", artifact_path)

    def register_model(
        self,
        run_id: Optional[str] = None,
        artifact_path: str = "model",
        model_name: str = "ad-ml-model",
        description: Optional[str] = None,
    ) -> str:
        """Register a model artifact to the MLflow Model Registry.

        Args:
            run_id: MLflow run ID containing the model artifact.
            artifact_path: Artifact path within the run.
            model_name: Registry model name.
            description: Optional description for the model version.

        Returns:
            The registered model version string.
        """
        run_id = run_id or self._active_run_id
        if run_id is None:
            raise RuntimeError("No active run. Call register_model inside a run() context.")

        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = registered.version

        if description:
            self._client.update_model_version(
                name=model_name, version=version, description=description
            )

        logger.info(
            "Registered model '%s' version %s from run %s",
            model_name, version, run_id,
        )
        return version

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> None:
        """Transition a registered model to a new stage.

        Args:
            model_name: Registry model name.
            version: Model version string.
            stage: Target stage: "Staging", "Production", or "Archived".
        """
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        logger.info("Transitioned model '%s' v%s to stage '%s'", model_name, version, stage)

    def compare_runs(
        self,
        metric_key: str,
        max_results: int = 10,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compare runs in this experiment sorted by a metric.

        Args:
            metric_key: Metric name to sort by.
            max_results: Maximum number of runs to return.
            ascending: Sort ascending (lower is better for loss metrics).

        Returns:
            List of run dicts with run_id, params, and the requested metric.
        """
        order = "ASC" if ascending else "DESC"
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string="",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=max_results,
            order_by=[f"metrics.{metric_key} {order}"],
        )
        results: List[Dict[str, Any]] = []
        for run in runs:
            results.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "params": dict(run.data.params),
                    metric_key: run.data.metrics.get(metric_key),
                    "status": run.info.status,
                }
            )
        return results

    def get_best_run(
        self, metric_key: str, ascending: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Return the best run by a given metric."""
        runs = self.compare_runs(metric_key, max_results=1, ascending=ascending)
        return runs[0] if runs else None

    def load_pytorch_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any:
        """Load a PyTorch model from the registry.

        Args:
            model_name: Registered model name.
            stage: Model stage to load ("Staging" or "Production").

        Returns:
            Loaded PyTorch model.
        """
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("Loaded model '%s' from stage '%s'", model_name, stage)
        return model

    @property
    def run_id(self) -> Optional[str]:
        return self._active_run_id or (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        )
