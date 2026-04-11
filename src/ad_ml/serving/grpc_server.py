"""gRPC inference server for user persona and autobidding models."""

from __future__ import annotations

import argparse
import logging
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub classes used when protobuf generated code is not available.
# In production, run `make proto` to generate the real stubs.
# ---------------------------------------------------------------------------
try:
    import grpc  # type: ignore[import]
    from ad_ml.serving import inference_pb2, inference_pb2_grpc  # type: ignore[import]
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False
    logger.warning(
        "grpc or generated proto stubs not available. "
        "Run `make proto` to generate them before starting the server."
    )


class ModelRegistry:
    """Manage loaded model instances with versioning and statistics."""

    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        model: Any,
        version: str = "latest",
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        self._models[name] = {
            "model": model,
            "version": version,
            "status": "ready",
            "loaded_at": int(time.time()),
            "metadata": metadata or {},
            "total_predictions": 0,
            "latency_sum_ms": 0.0,
            "latency_count": 0,
        }
        logger.info("Registered model '%s' version '%s'", name, version)

    def get(self, name: str) -> Optional[Any]:
        entry = self._models.get(name)
        return entry["model"] if entry else None

    def record_prediction(self, name: str, latency_ms: float, count: int = 1) -> None:
        if name in self._models:
            self._models[name]["total_predictions"] += count
            self._models[name]["latency_sum_ms"] += latency_ms
            self._models[name]["latency_count"] += 1

    def info(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._models.get(name)
        if entry is None:
            return None
        avg_latency = (
            entry["latency_sum_ms"] / entry["latency_count"]
            if entry["latency_count"] > 0
            else 0.0
        )
        return {
            "model_name": name,
            "version": entry["version"],
            "status": entry["status"],
            "loaded_at_unix": entry["loaded_at"],
            "metadata": entry["metadata"],
            "total_predictions": entry["total_predictions"],
            "avg_latency_ms": avg_latency,
        }

    def all_statuses(self) -> Dict[str, str]:
        return {name: v["status"] for name, v in self._models.items()}


def _features_to_vector(request: Any, expected_dim: int) -> np.ndarray:
    """Convert a PredictRequest to a dense float32 numpy array."""
    if len(request.feature_vector) == expected_dim:
        return np.array(list(request.feature_vector), dtype=np.float32)
    # Fall back to float_features map
    vec = np.zeros(expected_dim, dtype=np.float32)
    for i, (k, v) in enumerate(request.float_features.items()):
        if i < expected_dim:
            vec[i] = v
    return vec


class InferenceServicer:
    """gRPC service implementation for model inference.

    Handles Predict, BatchPredict, GetModelInfo, and HealthCheck RPCs.
    This class is designed to be mixed in with the generated gRPC servicer base
    when proto stubs are available.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        persona_feature_dims: tuple[int, int] = (64, 32),
        autobid_feature_dims: tuple[int, int] = (64, 64),
    ) -> None:
        self.registry = registry
        # (campaign_dim, context_dim) for autobid
        self.autobid_feature_dims = autobid_feature_dims
        # (seq_dim, user_dim) for persona
        self.persona_feature_dims = persona_feature_dims

    def Predict(self, request: Any, context: Any) -> Any:
        """Handle single-sample prediction request."""
        t0 = time.perf_counter()
        model_name: str = request.model_name
        model = self.registry.get(model_name)

        if model is None:
            if _GRPC_AVAILABLE:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model '{model_name}' not found")
                return inference_pb2.PredictResponse()
            return None

        try:
            predictions, named = self._run_single_inference(model_name, model, request)
        except Exception as exc:
            logger.exception("Inference error for model '%s': %s", model_name, exc)
            if _GRPC_AVAILABLE:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return inference_pb2.PredictResponse()
            return None

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.registry.record_prediction(model_name, latency_ms)

        if _GRPC_AVAILABLE:
            return inference_pb2.PredictResponse(
                request_id=request.request_id,
                predictions=predictions,
                named_predictions=named,
                latency_ms=latency_ms,
                model_version=self.registry.info(model_name)["version"],  # type: ignore[index]
                cache_hit=False,
            )
        return {"predictions": predictions, "named_predictions": named, "latency_ms": latency_ms}

    def BatchPredict(self, request: Any, context: Any) -> Any:
        """Handle batched prediction requests."""
        t0 = time.perf_counter()
        model_name: str = request.model_name
        model = self.registry.get(model_name)

        if model is None:
            if _GRPC_AVAILABLE:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model '{model_name}' not found")
                return inference_pb2.BatchPredictResponse()
            return None

        responses = []
        for sub_request in request.requests:
            sub_response = self.Predict(sub_request, context)
            responses.append(sub_response)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.registry.record_prediction(model_name, total_latency_ms, count=len(responses))

        if _GRPC_AVAILABLE:
            return inference_pb2.BatchPredictResponse(
                batch_id=request.batch_id,
                responses=responses,
                total_latency_ms=total_latency_ms,
                num_cache_hits=0,
            )
        return {"responses": responses, "total_latency_ms": total_latency_ms}

    def GetModelInfo(self, request: Any, context: Any) -> Any:
        """Return metadata for a registered model."""
        info = self.registry.info(request.model_name)
        if info is None:
            if _GRPC_AVAILABLE:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                return inference_pb2.ModelInfoResponse()
            return {}

        if _GRPC_AVAILABLE:
            return inference_pb2.ModelInfoResponse(
                model_name=info["model_name"],
                version=info["version"],
                status=info["status"],
                loaded_at_unix=info["loaded_at_unix"],
                metadata=info["metadata"],
                total_predictions=info["total_predictions"],
                avg_latency_ms=info["avg_latency_ms"],
            )
        return info

    def HealthCheck(self, request: Any, context: Any) -> Any:
        """Return health status of all registered models."""
        statuses = self.registry.all_statuses()
        healthy = all(s == "ready" for s in statuses.values())

        if _GRPC_AVAILABLE:
            return inference_pb2.HealthCheckResponse(
                healthy=healthy,
                message="OK" if healthy else "One or more models not ready",
                model_statuses=statuses,
            )
        return {"healthy": healthy, "model_statuses": statuses}

    def _run_single_inference(
        self, model_name: str, model: Any, request: Any
    ) -> tuple[List[float], Dict[str, float]]:
        """Dispatch to model-specific inference logic."""
        if model_name == "user_persona":
            return self._persona_inference(model, request)
        elif model_name == "autobid":
            return self._autobid_inference(model, request)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _persona_inference(
        self, model: Any, request: Any
    ) -> tuple[List[float], Dict[str, float]]:
        seq_dim, user_dim = self.persona_feature_dims
        feat = np.array(list(request.feature_vector), dtype=np.float32)
        seq_feat = feat[:seq_dim].reshape(1, 1, seq_dim)
        user_feat = feat[seq_dim : seq_dim + user_dim].reshape(1, user_dim)

        with torch.no_grad():
            seq_t = torch.from_numpy(seq_feat)
            uf_t = torch.from_numpy(user_feat)
            len_t = torch.tensor([1])
            output = model(seq_t, uf_t, len_t)
        emb = output.user_embedding[0].cpu().tolist()
        cluster = int(output.cluster_probs[0].argmax().item())
        return emb, {"cluster_assignment": float(cluster)}

    def _autobid_inference(
        self, model: Any, request: Any
    ) -> tuple[List[float], Dict[str, float]]:
        camp_dim, ctx_dim = self.autobid_feature_dims
        feat = np.array(list(request.feature_vector), dtype=np.float32)
        camp_feat = feat[:camp_dim].reshape(1, camp_dim)
        ctx_feat = feat[camp_dim : camp_dim + ctx_dim].reshape(1, ctx_dim)

        with torch.no_grad():
            output = model(
                torch.from_numpy(camp_feat),
                torch.from_numpy(ctx_feat),
            )
        bid = float(output.bid_multiplier[0].item())
        return [bid], {"bid_multiplier": bid}


def serve(
    persona_checkpoint: Optional[Path] = None,
    autobid_checkpoint: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 8,
    max_message_length: int = 10 * 1024 * 1024,
) -> None:
    """Start the gRPC inference server.

    Args:
        persona_checkpoint: Path to user persona model checkpoint.
        autobid_checkpoint: Path to autobid model checkpoint.
        host: Bind host.
        port: gRPC port.
        max_workers: Thread pool size.
        max_message_length: Max gRPC message size in bytes.
    """
    if not _GRPC_AVAILABLE:
        raise RuntimeError(
            "gRPC is not available. Install grpcio and run `make proto` to generate stubs."
        )

    registry = ModelRegistry()

    # Load models if checkpoints provided
    if persona_checkpoint and persona_checkpoint.exists():
        from ad_ml.models.user_persona.inference import PersonaInference
        persona = PersonaInference(checkpoint_path=persona_checkpoint)
        registry.register("user_persona", persona.model, metadata={"checkpoint": str(persona_checkpoint)})

    if autobid_checkpoint and autobid_checkpoint.exists():
        from ad_ml.models.autobid.inference import AutobidInference
        autobid = AutobidInference(checkpoint_path=autobid_checkpoint)
        registry.register("autobid", autobid.model, metadata={"checkpoint": str(autobid_checkpoint)})

    servicer = InferenceServicer(registry=registry)

    options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)  # type: ignore[attr-defined]

    bind_address = f"{host}:{port}"
    server.add_insecure_port(bind_address)
    server.start()
    logger.info("gRPC server started on %s with %d workers", bind_address, max_workers)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(grace=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ad ML gRPC inference server")
    parser.add_argument("--persona-checkpoint", type=Path, default=None)
    parser.add_argument("--autobid-checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    serve(
        persona_checkpoint=args.persona_checkpoint,
        autobid_checkpoint=args.autobid_checkpoint,
        host=args.host,
        port=args.port,
        max_workers=args.workers,
    )
