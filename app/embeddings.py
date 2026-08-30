"""
Embedding and semantic search module.

Uses Databricks foundation model endpoints to generate text embeddings.
Embeddings are stored as JSON-serialized float arrays in Lakebase TEXT
columns and compared with cosine similarity in Python.
"""

import json
import math
import os
import hashlib

# Try to use Databricks SDK for foundation model embeddings
try:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
    _USE_DATABRICKS = True
except Exception:
    _USE_DATABRICKS = False

_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "databricks-gte-large-en")
_EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")


def create_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    if not text or not text.strip():
        return []

    if _USE_DATABRICKS:
        try:
            return _databricks_embedding(text)
        except Exception:
            pass

    # Fallback: deterministic hash-based pseudo-embedding (384 dims)
    return _hash_embedding(text, 384)


def _databricks_embedding(text: str) -> list[float]:
    """Call Databricks foundation model for embeddings."""
    import mlflow.deployments  # type: ignore
    client = mlflow.deployments.get_deploy_client("databricks")
    resp = client.predict_endpoint(
        name=_EMBEDDING_ENDPOINT,
        inputs={"input": [text]},
    )
    # Response format: {"predictions": [{"embedding": [...]}]}
    predictions = resp.get("predictions", resp.get("data", []))
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            return first.get("embedding", first.get("data", []))
        return first
    return []


def _hash_embedding(text: str, dims: int = 384) -> list[float]:
    """Generate a deterministic pseudo-embedding from text using hashing.

    Not semantically meaningful but provides consistent vectors for
    similarity comparison when no embedding model is available.
    """
    vec = [0.0] * dims
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for i in range(min(4, dims)):
            idx = (h + i * 7919) % dims
            vec[idx] += ((h >> (i * 8)) & 0xFF) / 255.0
    # Normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def serialize_embedding(emb: list[float]) -> str:
    """Serialize an embedding vector for database storage."""
    return json.dumps(emb)


def deserialize_embedding(text: str) -> list[float]:
    """Deserialize an embedding vector from database storage."""
    try:
        return json.loads(text) if text else []
    except Exception:
        return []
