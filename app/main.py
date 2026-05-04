import time

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.config import settings
from app.ingest import ingest_documents
from app.retrieve import retrieve_context

app = FastAPI(title="RAG DataOps Pipeline", version="1.0.0")

REQUEST_COUNT = Counter(
    "rag_api_requests_total",
    "Total RAG API requests",
    ["endpoint", "environment"],
)

REQUEST_LATENCY = Histogram(
    "rag_api_request_latency_seconds",
    "RAG API request latency",
    ["endpoint", "environment"],
)


@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health", environment=settings.env_name).inc()
    return {
        "status": "ok",
        "environment": settings.env_name,
        "collection": settings.collection_name,
    }


@app.post("/ingest")
def ingest():
    start = time.time()
    result = ingest_documents()
    REQUEST_COUNT.labels(endpoint="/ingest", environment=settings.env_name).inc()
    REQUEST_LATENCY.labels(endpoint="/ingest", environment=settings.env_name).observe(
        time.time() - start
    )
    return result


@app.get("/ask")
def ask(question: str):
    start = time.time()
    context = retrieve_context(question)
    REQUEST_COUNT.labels(endpoint="/ask", environment=settings.env_name).inc()
    REQUEST_LATENCY.labels(endpoint="/ask", environment=settings.env_name).observe(
        time.time() - start
    )

    return {
        "environment": settings.env_name,
        "question": question,
        "answer": "This lab returns retrieved context only. Connect an LLM later for final answer generation.",
        "context": context,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
