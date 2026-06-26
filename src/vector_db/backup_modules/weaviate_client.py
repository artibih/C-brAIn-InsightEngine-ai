"""Shared Weaviate client for scripts. Matches WeaviateDB connection (localhost or env)."""

import os
import weaviate


def get_client():
    """Return Weaviate client: local or custom from WEAVIATE_* env."""
    host = os.environ.get("WEAVIATE_HOST", "localhost")
    if host != "localhost":
        http_secure = os.environ.get("WEAVIATE_HTTP_SECURE", "False").strip().lower() == "true"
        grpc_secure = os.environ.get("WEAVIATE_GRPC_SECURE", "False").strip().lower() == "true"
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=int(os.environ.get("WEAVIATE_PORT", "8080")),
            http_secure=http_secure,
            grpc_host=os.environ.get("WEAVIATE_GRPC_HOST", host),
            grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
            grpc_secure=grpc_secure,
        )
    return weaviate.connect_to_local(
        host=host,
        port=int(os.environ.get("WEAVIATE_PORT", "8080")),
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
    )
