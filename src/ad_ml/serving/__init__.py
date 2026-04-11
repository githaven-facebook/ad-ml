"""gRPC inference serving."""

from ad_ml.serving.grpc_server import InferenceServicer, serve

__all__ = ["InferenceServicer", "serve"]
