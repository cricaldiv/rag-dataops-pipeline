# RAG DataOps CI/CD Lab

This lab implements a simple RAG-style DataOps pipeline with TEST and PROD environments.

## Stack

- FastAPI RAG API
- Qdrant vector database
- SentenceTransformers embeddings
- Docker Compose
- GitHub Actions CI/CD

## Local TEST

```bash
docker compose -f docker-compose.test.yml up -d --build
curl -X POST http://localhost:8001/ingest
curl "http://localhost:8001/ask?question=what%20stores%20logs"
```

## Local PROD

```bash
docker compose -f docker-compose.prod.yml up -d --build
curl -X POST http://localhost:8002/ingest
curl "http://localhost:8002/ask?question=what%20requires%20approval%20before%20production"
```

## DataOps Controls

- Documents are version-controlled.
- TEST and PROD use separate vector DB collections.
- TEST uses sample documents.
- PROD uses approved documents.
- Promotion to PROD requires GitHub Environment approval.
- Smoke tests validate retrieval after ingestion.

## Interview explanation

I designed a DataOps-oriented RAG pipeline where code, documents, ingestion logic, and configuration are version-controlled. The CI/CD process validates code, builds the container image, deploys to TEST, ingests test documents, runs retrieval smoke tests, and promotes to PROD only after approval. TEST and PROD use separate vector databases and collections to avoid environment contamination.
