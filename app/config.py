from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env_name: str = "test"
    qdrant_url: str = "http://qdrant-test:6333"
    collection_name: str = "rag_test"
    docs_path: str = "data/test/sample_docs.txt"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_limit: int = 3

    class Config:
        env_prefix = "RAG_"


settings = Settings()
