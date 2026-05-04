from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def retrieve_context(question: str) -> list[dict]:
    model = get_model()
    vector = model.encode(question).tolist()
    client = get_client()

    results = client.search(
        collection_name=settings.collection_name,
        query_vector=vector,
        limit=settings.similarity_limit,
    )

    return [
        {
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", "unknown"),
            "env": hit.payload.get("env", settings.env_name),
        }
        for hit in results
    ]
