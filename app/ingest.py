from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings


def load_documents(path: str) -> list[str]:
    content = Path(path).read_text(encoding="utf-8")
    docs = [block.strip() for block in content.split("\n---\n") if block.strip()]
    if not docs:
        raise ValueError(f"No valid documents found in {path}")
    return docs


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    collections = client.get_collections().collections
    existing = {collection.name for collection in collections}

    if settings.collection_name not in existing:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def ingest_documents() -> dict:
    docs = load_documents(settings.docs_path)
    model = SentenceTransformer(settings.embedding_model)
    client = QdrantClient(url=settings.qdrant_url)

    sample_vector = model.encode("health check").tolist()
    ensure_collection(client, len(sample_vector))

    points = []
    for doc in docs:
        vector = model.encode(doc).tolist()
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": doc,
                    "source": settings.docs_path,
                    "env": settings.env_name,
                },
            )
        )

    client.upsert(collection_name=settings.collection_name, points=points)

    return {
        "environment": settings.env_name,
        "collection": settings.collection_name,
        "documents_ingested": len(points),
    }


if __name__ == "__main__":
    print(ingest_documents())
