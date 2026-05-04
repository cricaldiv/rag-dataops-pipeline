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
        unique_texts = set()
        filtered_results = []

        for hit in results:
            text = hit.payload.get("text", "")
            if text not in unique_texts:
                unique_texts.add(text)
                filtered_results.append({
                    "score": hit.score,
                    "text": text,
                    "source": hit.payload.get("source", "unknown"),
                    "env": hit.payload.get("env", settings.env_name),
                })

        return filtered_results
    ]
